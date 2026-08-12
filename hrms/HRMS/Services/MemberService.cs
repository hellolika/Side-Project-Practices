using System.Collections.Generic;
using System.Data;
using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Repositories.Interfaces;
using HRMS.Services.Interfaces;

namespace HRMS.Services
{
    public class MemberService : IMemberService
    {
        private readonly ISha512Service _sha512Service;
        private readonly IMemberRepository _memberRepository;
        private readonly IWorkingDateService _workingDateService;
        private readonly IMemberDataService _memberDataService;
        private readonly IImageService _imageService;
        private readonly IBackOfficeRepository _backOfficeRepository;

        public MemberService(ISha512Service sha512Service, IMemberRepository memberRepository,
            IWorkingDateService workingDateService, IMemberDataService memberDataService, IImageService imageService,
            IBackOfficeRepository backOfficeRepository)
        {
            _sha512Service = sha512Service;
            _memberRepository = memberRepository;
            _workingDateService = workingDateService;
            _memberDataService = memberDataService;
            _imageService = imageService;
            _backOfficeRepository = backOfficeRepository;
        }

        public ApiBaseResponse<string> DoClock(DoClockRequest request)
        {
            request.CheckModels();
            request.MemberId = _memberDataService.GetCurrentMemberId();

            request.Date = DateTimeOffset.Now.ToOffset(request.TimeZone.Value);
            var response = _memberRepository.DoClock(request);

            response.CheckErrorCode();

            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<ClockStatusResponse> CheckClockStatus(ClockStatusRequest request)
        {
            request.CheckModels();
            request.MemberId = _memberDataService.GetCurrentMemberId();
            return new ApiBaseResponse<ClockStatusResponse>
                (_memberRepository.CheckClockStatus(request));
        }

        public ApiBaseResponse<SubmitFormBaseResponse> DoReclock(DoReClockRequest request)
        {
            request.CheckModels();
            request.MemberId = _memberDataService.GetCurrentMemberId();
            var response = _memberRepository.DoReclock(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<SubmitFormBaseResponse>(response);
        }

        public ApiBaseResponse<SubmitFormBaseResponse> TakeLeave(TakeLeaveRequest request)
        {
            var numberOfLeaveDays = _workingDateService.GetNumberOfLeavesDays(request.StartDate, request.EndDate);
            request.CheckNumberOfDays(numberOfLeaveDays);
            request.CheckModels();
            request.MemberId = _memberDataService.GetCurrentMemberId();
            if (request.RequestId != 0)
            {
                var response = _memberRepository.UpdateTakeLeave(request);
                response.CheckErrorCode();
                return new ApiBaseResponse<SubmitFormBaseResponse>(response);
            }
            else
            {
                var leavePeriods = _workingDateService.GetLeavePeriods(request.StartDate, request.EndDate);
                SubmitFormBaseResponse response = null;
                foreach (var leavePeriod in leavePeriods)
                {
                    request.NumberOfDay =
                        _workingDateService.GetNumberOfLeavesDays(leavePeriod.StartDate, leavePeriod.EndDate);
                    request.StartDate = leavePeriod.StartDate;
                    request.EndDate = leavePeriod.EndDate;
                    response = _memberRepository.TakeLeave(request);
                    response.CheckErrorCode();
                }

                return new ApiBaseResponse<SubmitFormBaseResponse>(response);
            }
        }

        public ApiBaseResponse<List<TransferResponse>> GetTransfers()
        {
            var memberId = _memberDataService.GetCurrentMemberId();
            var transfers = _memberRepository.GetTransfers(memberId);
            var monthlyTransfers = transfers.Where(e => e.TransferTypeId == 1).ToList();

            foreach (var i in monthlyTransfers)
            {
                var listDeduction = transfers.Where(m =>
                    m.MemberId == i.MemberId && m.PayDate.ToString("MMMM") == i.PayDate.ToString("MMMM") &&
                    m.BeneficiaryTypeId == Convert.ToInt32(BeneficiaryTypeEnum.Deduction)).ToList();
                var listAllowance = transfers.Where(m =>
                    m.MemberId == i.MemberId && m.PayDate.ToString("MMMM") == i.PayDate.ToString("MMMM") &&
                    m.BeneficiaryTypeId == Convert.ToInt32(BeneficiaryTypeEnum.Allowance)).ToList();
                var deductionList = new List<BeneficiaryType>();
                var allowanceList = new List<BeneficiaryType>();
                double absentCount = 0;
                double totalDeductionAmount = 0;
                double totalAllowanceAmount = 0;

                foreach (var r in listDeduction)
                {
                    var deduction = new BeneficiaryType()
                    {
                        TransferTypeId = r.TransferTypeId,
                        TransferId = r.TransferId,
                        Name = r.TransferName,
                        Amount = r.Amount,
                        DayCount = r.DateCount,
                        Remark = r.Remark,
                        ModifiedBy = r.ModifiedBy,
                        Modifier = r.Modifier,
                        BeneficiaryTypeId = r.BeneficiaryTypeId
                    };
                    if (r.TransferTypeId == Convert.ToInt32(TransferTypeEnum.UnpaidLeave) && r.IsGenerated)
                        deduction.TakeLeaveRecords = _memberRepository
                            .GetMemberTakeLeaveRecords(new MemberTimeTableRequest()
                                {MemberId = i.MemberId, StartDate = r.PayStartDate, EndDate = r.PayEndDate})
                            .Where(l => l.LeaveId == 0).ToList();
                    absentCount = r.DateCount;
                    deductionList.Add(deduction);
                    totalDeductionAmount += r.Amount;
                }

                foreach (var r in listAllowance)
                {
                    var allowance = new BeneficiaryType()
                    {
                        TransferTypeId = r.TransferTypeId,
                        TransferId = r.TransferId,
                        Name = r.TransferName,
                        Amount = r.Amount,
                        DayCount = r.DateCount,
                        Remark = r.Remark,
                        ModifiedBy = r.ModifiedBy,
                        Modifier = r.Modifier,
                        BeneficiaryTypeId = r.BeneficiaryTypeId
                    };
                    // absentCount = r.DateCount;
                    allowanceList.Add(allowance);
                    totalAllowanceAmount += r.Amount;
                }

                i.Deduction = deductionList;
                i.Allowance = allowanceList;
                i.TotalAllowanceAmount = totalAllowanceAmount;
                i.TotalDeductionAmount = totalDeductionAmount;
                i.TotalBeneficiaryAmount = totalAllowanceAmount - totalDeductionAmount;
                i.WorkDay = i.DateCount == 0 ? (i.PayEndDate - i.PayStartDate).TotalDays : i.DateCount;
                i.Absent = absentCount;
                i.NetSalary = (i.DateCount > 0 ? (i.Amount / 30) * i.DateCount : i.Amount) + i.TotalBeneficiaryAmount;
            }

            return new ApiBaseResponse<List<TransferResponse>>(monthlyTransfers);
        }

        public async Task<ApiBaseResponse<UploadImageResponse>> UploadLeaveImage(UploadImageRequest request)
        {
            if (!await _imageService.CheckImageSize(request.FormFile))
            {
                return new ApiBaseResponse<UploadImageResponse>(ApiErrorEnum.ImageSizeLimitMb);
            }

            request.Folder = "HRMS_Leave";
            var response = await _imageService.UploadImage(request);
            return response;
        }

        public ApiBaseResponse<string> CancelLeave(CancelLeaveRequest request)
        {
            request.MemberId = _memberDataService.GetCurrentMemberId();
            var response = _memberRepository.CancelLeave(request);

            response.CheckErrorCode();

            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<List<TimeTableResponse>> GetTimeTable(TimeTableRequest request)
        {
            var memberId = _memberDataService.GetCurrentMemberId();
            var memberTimeTable = new MemberTimeTableRequest(request, memberId);
            var timeTableRecords = new List<TimeTableResponse>();

            var attendances = _memberRepository.GetMemberAttendances(memberTimeTable);
            var reclocks = _memberRepository.GetMemberReClockRecords(memberTimeTable);
            var leaves = _memberRepository.GetMemberTakeLeaveRecords(memberTimeTable);

            var clientOffSet = request.StartDate.Offset;

            for (DateTime date = request.StartDate.Date; date <= request.EndDate.Date; date = date.AddDays(1))
            {
                if (_workingDateService.IsWorkingDate(date))
                {
                    var timeTable = new TimeTableResponse(date, clientOffSet);

                    if (attendances.FirstOrDefault(a => a.WorkDate == date) is Attendance attentance)
                    {
                        timeTable.FillAttendance(attentance);
                    }

                    foreach (var reclock in reclocks.Where(r => r.Date == date))
                    {
                        timeTable.FillReClockRecord(reclock);
                    }

                    if (leaves.FirstOrDefault(l => l.StartDate.Date <= date && l.EndDate.Date >= date && !l.IsCancel)
                        is TakeLeaveRecord takeLeave)
                    {
                        timeTable.FillLeaveRecord(takeLeave);
                    }

                    timeTableRecords.Add(timeTable);
                }
            }

            return new ApiBaseResponse<List<TimeTableResponse>>(timeTableRecords);
        }

        public ApiBaseResponse<List<LeaveAmountResponse>> GetMemberLeaveAmount()
        {
            var memberId = _memberDataService.GetCurrentMemberId();
            var leaveAmount = _memberRepository.GetMemberLeaveAmount(memberId);

            return new ApiBaseResponse<List<LeaveAmountResponse>>(leaveAmount);
        }

        public ApiBaseResponse<List<LeaveAmountResponseV2>> GetMemberLeaveAmountV2(int memberId = 0)
        {
            memberId = memberId.Equals(0) ? _memberDataService.GetCurrentMemberId() : memberId;
            var leaveAmount = _memberRepository.GetMemberLeaveAmountV2(memberId);

            if (leaveAmount.Count.Equals(0))
            {
                throw new ApiException(ApiErrorEnum.MemberHaveNoLeaveAmount,
                    "This member does not have any leave amount or doesn't exist");
            }

            return new ApiBaseResponse<List<LeaveAmountResponseV2>>(leaveAmount);
        }

        public ApiBaseResponse<AllRequestFormsResponse> GetAllRequestForms(BaseRequest request)
        {
            request.CheckModels();

            var memberId = _memberDataService.GetCurrentMemberId();
            var takeLeaveForms = _memberRepository.GetMemberLeaveRequestRecords(memberId);
            var reClockForms = _memberRepository.GetAllRequestFormsReClock(memberId);

            foreach (var reclock in reClockForms)
            {
                reclock.Time = reclock.Time.Add(request.TimeZone.Value);
            }

            var forms = new AllRequestFormsResponse(takeLeaveForms, reClockForms);

            return new ApiBaseResponse<AllRequestFormsResponse>(forms);
        }

        public ApiBaseResponse<MemberProfile> GetProfile(int memberId, bool isAdmin, int jwtMemberId)
        {
            memberId = memberId.Equals(0) ? _memberDataService.GetCurrentMemberId() : memberId;
            var permissionResponse = _memberRepository.GetPermissionsByMemberId(memberId);
            List<int> memberPermission = permissionResponse.Select(p => p.PermissionId).ToList();
            List<GetAllPermissionResponse> permission = new List<GetAllPermissionResponse> { };

            foreach (var i in permissionResponse)
            {
                var permissionRes = new GetAllPermissionResponse()
                {
                    Id = i.PermissionId,
                    PermissionName = i.PermissionName
                };
                permission.Add(permissionRes);
            }

            //Checking current user permission
            var currentUserProfile = _memberRepository.GetProfile(jwtMemberId);
            
            var profile = _memberRepository.GetProfile(memberId);
            profile.CheckErrorCode();
            profile.Permissions = permission.Distinct().ToList();
            
            if (!IsAuthorizedToViewSalary(currentUserProfile, isAdmin))
            {
                profile.Salary = 0;
            }
            
            return new ApiBaseResponse<MemberProfile>(profile);
        }

        public bool IsAuthorizedToViewSalary(MemberProfile profile, bool isAdmin)
        {
            return isAdmin || profile.IsCanSeeMemberSalary;
        }

        public ApiBaseResponse<int> UpdateProfile(UpdateProfileRequest request)
        {
            request.CheckModel();
            var memberId = _memberDataService.GetCurrentMemberId();
            var response = _memberRepository.UpdateProfile(request, memberId);

            response.CheckErrorCode();

            return new ApiBaseResponse<int>(0);
        }

        public ApiBaseResponse<List<LeaveType>> GetAllLeaveType()
        {
            return new ApiBaseResponse<List<LeaveType>>(_memberRepository.GetAllLeaveType());
        }

        public ApiBaseResponse<List<TeamInfo>> GetAllTeamInfo()
        {
            return new ApiBaseResponse<List<TeamInfo>>(_memberRepository.GetAllTeamInfo());
        }

        public ApiBaseResponse<List<LocationDetails>> GetLocation()
        {
            return new ApiBaseResponse<List<LocationDetails>>(_memberRepository.GetLocation());
        }

        public ApiBaseResponse<string> ChangePassword(ChangePasswordRequest request)
        {
            request.CheckModels();

            request.MemberId = _memberDataService.GetCurrentMemberId();
            request.Password = _sha512Service.Encrypt(request.Password);
            request.NewPassword = _sha512Service.Encrypt(request.NewPassword);

            var response = _memberRepository.ChangePassword(request);

            response.CheckErrorCode();

            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<GetAnnouncementResponse> GetAnnouncements(GetAnnouncementRequest request)
        {
            request.CheckModels();
            var total = _memberRepository.GetTotalAnnouncements();
            var announcements = _memberRepository.GetAnnouncements(request);
            foreach (var announcement in announcements)
            {
                announcement.CreatedOn = announcement.CreatedOn.ToOffset(request.TimeZone.Value);
                announcement.ModifiedOn = announcement.ModifiedOn.ToOffset(request.TimeZone.Value);
            }

            var reponse = new GetAnnouncementResponse()
            {
                Announcements = announcements,
                CurrentPage = request.Page,
                ItemPerPage = request.ItemPerPage,
                TotalPages = (int) Math.Ceiling(decimal.Divide(total, request.ItemPerPage))
            };

            return new ApiBaseResponse<GetAnnouncementResponse>(reponse);
        }
        
        public ApiBaseResponse<GetAvailableSettingForMemberOperationResponse> GetAvailableSettingForMemberOperation()
        {
            var response = new GetAvailableSettingForMemberOperationResponse();
            response.Departments = _backOfficeRepository.GetAllDepartments();
            response.Teams = _memberRepository.GetAllTeamInfo();
            response.Positions = _backOfficeRepository.GetAllPosition();
            response.JobGrades = _backOfficeRepository.GetAllJobGrade();
            return new ApiBaseResponse<GetAvailableSettingForMemberOperationResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> UpsertSlackUsers(List<SlackUser> slackUsers)
        {
            var slackUserDataTable = new DataTable();
            slackUserDataTable.Columns.Add("Id", typeof(string));
            slackUserDataTable.Columns.Add("Username", typeof(string));
            slackUserDataTable.Columns.Add("RealName", typeof(string));
            slackUserDataTable.Columns.Add("Email", typeof(string));
            foreach (var slackUser in slackUsers)
            {
                var slackUserRow = slackUserDataTable.NewRow();
                slackUserRow["Id"] = slackUser.Id;
                slackUserRow["Username"] = slackUser.Name;
                slackUserRow["RealName"] = slackUser.RealName;
                slackUserRow["Email"] = slackUser.Profile.Email;
                slackUserDataTable.Rows.Add(slackUserRow);
                
            }
            return new ApiBaseResponse<RepositoryBaseResponse>(_memberRepository.UpsertSlackUsers(slackUserDataTable));
        }
    }
}