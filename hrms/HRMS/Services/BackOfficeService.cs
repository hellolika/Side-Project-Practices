using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Repositories.Interfaces;
using HRMS.Services.Interfaces;
using System.Data;
using System.Linq;


namespace HRMS.Services
{
    public class BackOfficeService : IBackOfficeService
    {
        private readonly IBackOfficeRepository _backOfficeRepository;
        private readonly IWorkingDateService _workingDateService;
        private readonly IMemberDataService _memberDataService;
        private readonly ISha512Service _sha512Service;
        private readonly IMemberRepository _memberRepository;
        private readonly INotificationService _notificationService;
        private readonly ISlackService _slackService;

        public BackOfficeService(IBackOfficeRepository backOfficeRepository, ISha512Service sha512Service,
            IWorkingDateService workingDateService, IMemberDataService memberDataService,
            IMemberRepository memberRepository, INotificationService notificationService, ISlackService slackService)
        {
            _backOfficeRepository = backOfficeRepository;
            _sha512Service = sha512Service;
            _workingDateService = workingDateService;
            _memberDataService = memberDataService;
            _memberRepository = memberRepository;
            _notificationService = notificationService;
            _slackService = slackService;
        }

        public ApiBaseResponse<List<GetMembersResponse>> GetMembers(JwtData jwtData)
        {
            var response = _backOfficeRepository.GetMembers();
            
            if (jwtData != null)
            {
                var profile = _memberRepository.GetProfile(jwtData.Id);
                var canSeeMemberSalary = jwtData.IsAdmin() || profile.IsCanSeeMemberSalary;
                if (!canSeeMemberSalary)
                {
                    foreach (var member in response)
                    {
                        member.Salary = 0;
                    }
                }
            }
            
            return new ApiBaseResponse<List<GetMembersResponse>>(response);
        }

        public ApiBaseResponse<List<LeavesResponse>> GetAllLeaveRequests(TimeRangeRequest request)
        {
            return new ApiBaseResponse<List<LeavesResponse>>(_backOfficeRepository.GetAllLeaves(request));
        }

        public ApiBaseResponse<List<ReClocksResponse>> GetAllReClockRequests(TimeRangeRequest request)
        {
            return new ApiBaseResponse<List<ReClocksResponse>>(_backOfficeRepository.GetAllReClock(request));
        }

        public ApiBaseResponse<List<GetAbsenteeResponse>> GetAllAbsentee(GetAbsenteeRequest request)
        {
            var workingDays = _workingDateService.GetWorkingDays(request.StartDate.Date, request.EndDate.Date);
            var tvpWorkingDay = CreateDataTable(workingDays);
            var response = _backOfficeRepository.GetAbsentee(tvpWorkingDay).ToList();

            return new ApiBaseResponse<List<GetAbsenteeResponse>>(response);
        }

        public ApiBaseResponse<string> LeaveRequestApproval(RequestApprovalRequest request)
        {
            request.CheckModels();
            request.ApproverId = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.LeaveRequestApproval(request);
            response.CheckErrorCode();
            if (response.ErrorCode == ApiErrorEnum.NoError)
            {
                LeaveApprovalNotificationById(request);
            }

            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> ReClockRequestApproval(RequestApprovalRequest request)
        {
            request.CheckModels();
            request.ApproverId = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.ReClockRequestApproval(request);
            response.CheckErrorCode();

            return new ApiBaseResponse<string>(null);
        }

        public ApiBaseResponse<string> AddNewLocation(AddLocationRequest location)
        {
            var response = _backOfficeRepository.AddNewLocation(location);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> RegisterMember(RegisterMemberRequest request)
        {
            request.CheckModels();
            request.Password = _sha512Service.Encrypt(request.Password);
            var createdBy = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.RegisterMember(request);
            response.CheckErrorCode();

            if (!request.IsInProbation)
            {
                var registeredMemberId = response.MemberId;
                var byMemberIdRequest = new ByMemberIdRequest {MemberId = registeredMemberId};
                _backOfficeRepository.RegisterLeaveAmount(byMemberIdRequest);
            }

            var unpaidLeaveResponse = _backOfficeRepository.AddUnpaidLeave(response.MemberId);
            unpaidLeaveResponse.CheckErrorCode();
            _backOfficeRepository.AddMemberRole(new AddMemberRoleRequest()
                {MemberId = response.MemberId, RoleId = Convert.ToInt32(RoleTypeEnum.Staff), CreatedBy = createdBy});
            
            // sync slack user to db
            Task.Run(() =>
            {
                _slackService.GetAllSlackUsers();
            });
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> EditMember(EditMemberInfoRequest request)
        {
            request.CheckModels();
            var response = _backOfficeRepository.EditMember(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> DeleteMember(DeleteMemberRequest request)
        {
            var response = _backOfficeRepository.DeleteMember(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> ResetPassword(PasswordBaseRequest request)
        {
            request.NewPassword = _sha512Service.Encrypt(request.NewPassword);
            var response = _backOfficeRepository.ResetPassword(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<List<GetAllLeaveAmountResponse>> GetAllLeaveAmount()
        {
            var allLeaves = _backOfficeRepository.GetAllLeaveAmount();
            var groupLeaves = allLeaves.GroupBy(l => new {l.MemberId, l.Year, l.Username})
                .Select(g => new GetAllLeaveAmountResponse(g.Key.MemberId, g.Key.Username, g.Key.Year, g.ToList()))
                .ToList();

            return new ApiBaseResponse<List<GetAllLeaveAmountResponse>>(groupLeaves);
        }

        public ApiBaseResponse<string> UpdateLeaveAmount(UpdateLeaveAmountRequest request)
        {
            request.ModifierId = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.UpdateLeaveAmount(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> PopulateDefaultLeaves()
        {
            var response = _backOfficeRepository.PopulateDefaultLeaves();
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> PopulateMemberRoleRecord(PopulateMemberRoleRecordRequest request)
        {
            var response = _backOfficeRepository.PopulateMemberRoleRecord(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> AddAnnouncement(AddAnnouncementRequest request)
        {
            request.CheckModels();
            request.CreatedBy = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.AddAnnouncement(request);
            response.CheckErrorCode();
            if (response.ErrorCode == ApiErrorEnum.NoError)
            {
                var notificationRequest = new SendNotificationToAllSubscriberRequest();
                var notificationMessage = new NotificationMessage()
                {
                    Title = request.Title,
                    Content = request.Message,
                };
                notificationRequest.Messages = new List<NotificationMessage> {notificationMessage};

                _notificationService.SendNotificationToAllSubscribers(notificationRequest);
            }

            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> EditAnnouncement(EditAnnouncementRequest request)
        {
            request.CheckModels();
            request.ModifiedBy = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.EditAnnouncement(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> DeleteAnnouncement(int id)
        {
            if (id <= 0)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Id is required");
            }

            var response = _backOfficeRepository.DeleteAnnouncement(id);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> AddTransferType(AddTransferTypeRequest request)
        {
            var response = _backOfficeRepository.AddTransferType(request);
            request.CreateBy = _memberDataService.GetCurrentMemberId();
            request.ModifiedBy = _memberDataService.GetCurrentMemberId();

            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<List<TransferTypeResponse>> GetAllTransferTypes()
        {
            var allTransferTypes = _backOfficeRepository.GetAllTransferTypes();
            return new ApiBaseResponse<List<TransferTypeResponse>>(allTransferTypes);
        }

        public ApiBaseResponse<List<TransferResponse>> GenerateMemberMonthlyTransfer(
            GenerateMemberMonthlyTransferRequest request)
        {
            var monthlyTransfers = _backOfficeRepository.GenerateMemberMonthlyTransfer(request);
            return new ApiBaseResponse<List<TransferResponse>>(monthlyTransfers);
        }

        public ApiBaseResponse<string> AddMonthlyTransfer(AddMonthlyTransferRequest request)
        {
            request.CreateBy = _memberDataService.GetCurrentMemberId();
            request.ModifiedBy = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.AddMonthlyTransfer(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> DeleteMonthlyTransfer(DeleteMonthlyTransferRequest request)
        {
            var response = _backOfficeRepository.DeleteMonthlyTransfer(request);

            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        // NEED REFACTOR
        public ApiBaseResponse<GetMontlyTransferResponse> GetMonthlyTransfers(
            GenerateMemberMonthlyTransferRequest request, bool isAdmin)
        {
            var memberId = _memberDataService.GetCurrentMemberId();
            var profile = _memberRepository.GetProfile(memberId);
            var canSeeMemberSalary = profile.IsCanSeeMemberSalary || isAdmin;
            var monthlyTransfers = _backOfficeRepository.GetMonthlyTransfers(request);
            var beneficiaryTypes = _backOfficeRepository.GetBeneficiaryType();
            var transferType = _backOfficeRepository.GetAllTransferTypes();
            foreach (var b in beneficiaryTypes)
            {
                b.BeneficiaryTypes = transferType.Where(t => t.BeneficiaryTypeId == b.Id).ToList();
            }

            foreach (var i in monthlyTransfers)
            {
                var listDeduction = monthlyTransfers.Where(m =>
                        m.MemberId == i.MemberId &&
                        m.BeneficiaryTypeId == Convert.ToInt32(BeneficiaryTypeEnum.Deduction))
                    .ToList();
                var listAllowance = monthlyTransfers.Where(m =>
                        m.MemberId == i.MemberId &&
                        m.BeneficiaryTypeId == Convert.ToInt32(BeneficiaryTypeEnum.Allowance))
                    .ToList();
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
                i.Amount =  canSeeMemberSalary ? i.Amount : 0;
                i.BeneficiaryTypes = beneficiaryTypes;
                i.Deduction = deductionList;
                i.Allowance = allowanceList;
                i.TotalAllowanceAmount =  canSeeMemberSalary ? totalAllowanceAmount : 0;
                i.TotalDeductionAmount =  canSeeMemberSalary ? totalDeductionAmount : 0;
                i.TotalBeneficiaryAmount =  canSeeMemberSalary ? totalAllowanceAmount - totalDeductionAmount : 0;
                i.WorkDay = i.DateCount == 0 ? (i.PayEndDate - i.PayStartDate).TotalDays : i.DateCount;
                i.Absent = absentCount;
                i.NetSalary = (i.DateCount > 0 ? (i.Amount / 30) * i.DateCount : i.Amount) + i.TotalBeneficiaryAmount;
            }

            var getOnlyMonthlyTransfer = monthlyTransfers.Where(m => m.TransferTypeId == 1).ToList();
            double totalNetSalary =  canSeeMemberSalary ? getOnlyMonthlyTransfer.Sum(item => (double) item.NetSalary) : 0;
            return new ApiBaseResponse<GetMontlyTransferResponse>(new GetMontlyTransferResponse()
                {Total = totalNetSalary, TransferListResponse = getOnlyMonthlyTransfer});
        }

        public ApiBaseResponse<SubmitFormBaseResponse> AddOrEditLeaveRecord(AddOrEditLeaveRecordRequest request)
        {
            request.CheckModels();
            var takeLeaveRequest = new TakeLeaveRequest()
            {
                RequestId = request.LeaveRecordId,
                MemberId = request.MemberId,
                NumberOfDay = request.NumberOfDay,
                StartDate = request.StartDate,
                EndDate = request.EndDate,
                LeaveType = request.LeaveType,
                ImagePath = request.ImagePath,
                Reason = "",
            };

            var numberOfLeaveDays = _workingDateService.GetNumberOfLeavesDays(request.StartDate, request.EndDate);
            takeLeaveRequest.CheckNumberOfDays(numberOfLeaveDays);
            takeLeaveRequest.CheckModels(checkReason: false);
            SubmitFormBaseResponse response = new SubmitFormBaseResponse() {ErrorCode = ApiErrorEnum.InvalidRequest};
            if (request.LeaveRecordId != 0)
            {
                response = _memberRepository.UpdateTakeLeave(takeLeaveRequest);
                ProceedLeaveRecordOperation(request, response);
            }
            else
            {
                var leavePeriods = _workingDateService.GetLeavePeriods(request.StartDate, request.EndDate);
                foreach (var leavePeriod in leavePeriods)
                {
                    takeLeaveRequest.NumberOfDay =
                        _workingDateService.GetNumberOfLeavesDays(leavePeriod.StartDate, leavePeriod.EndDate);
                    takeLeaveRequest.StartDate = leavePeriod.StartDate;
                    takeLeaveRequest.EndDate = leavePeriod.EndDate;
                    response = _memberRepository.TakeLeave(takeLeaveRequest);
                    response.CheckErrorCode();
                    ProceedLeaveRecordOperation(request, response);
                }
            }
            return new ApiBaseResponse<SubmitFormBaseResponse>(response);
        }

        private void ProceedLeaveRecordOperation(AddOrEditLeaveRecordRequest request, SubmitFormBaseResponse response)
        {
            if (response.ErrorCode == ApiErrorEnum.NoError && request.Status > 0)
            {
                var leaveRequestApprove = new RequestApprovalRequest()
                {
                    RequestId = response.RequestId,
                    MemberId = request.MemberId,
                    IsApproved = request.Status,
                    ResponseReason = request.Reason
                };
                LeaveRequestApproval(leaveRequestApprove);
            }
        }


        public ApiBaseResponse<string> EditMemberLeaves(EditMemberLeaveRequest request)
        {
            request.CheckModels();

            var response = _backOfficeRepository.EditMemberLeaves(request);
            response.CheckErrorCode();

            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<List<LeaveAmountResponse>> GetLeaveAmountByMemberId(
            GetLeaveAmountByMemberIdRequest request)
        {
            request.CheckModels();

            var response = _backOfficeRepository.GetLeaveAmountByMemberId(request);


            return new ApiBaseResponse<List<LeaveAmountResponse>>(response);
        }

        public ApiBaseResponse<List<MemberAttendance>> GetAllAttendance(GetAllAttendanceRequest request)
        {
            request.CheckModels();
            var response = _backOfficeRepository.GetAllAttendance(request);
            var afterGroup = response.GroupBy(x => x.MemberId).Select(x => x.First()).ToList();
            return new ApiBaseResponse<List<MemberAttendance>>(afterGroup);
        }


        public ApiBaseResponse<List<RoleResponse>> GetAllRole()
        {
            var allRole = _backOfficeRepository.GetAllRole();

            foreach (RoleResponse i in allRole)
            {
                var memberByRoleId = _backOfficeRepository.GetMemberByRoleId(i.Id);
                i.Members = memberByRoleId;
            }

            return new ApiBaseResponse<List<RoleResponse>>(allRole);
        }

        public ApiBaseResponse<string> AddMemberRole(AddMemberRoleRequest request)
        {
            request.CreatedBy = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.AddMemberRole(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> DeleteMemberRole(DeleteMemberRoleRequest request)
        {
            var response = _backOfficeRepository.DeleteMemberRole(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<DashboardResponse> GetDashboard()
        {
            var response = _backOfficeRepository.GetDashboard();
            response.CheckErrorCode();

            return new ApiBaseResponse<DashboardResponse>(response);
        }

        public ApiBaseResponse<Dictionary<string, List<GetAllPermissionResponse>>> GetAllPermission()
        {
            var response = _backOfficeRepository.GetAllPermission();
            var groupByCategory = response.GroupBy(s => s.PermissionCategoryName);
            var dictionary = new Dictionary<string, List<GetAllPermissionResponse>>();
            foreach (var group in groupByCategory)
            {
                dictionary.Add(group.Key, group.ToList());
            }

            return new ApiBaseResponse<Dictionary<string, List<GetAllPermissionResponse>>>(dictionary);
        }

        public ApiBaseResponse<string> AddRole(AddRoleRequest request)
        {
            request.CreatedBy = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.AddRole(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<List<int>> GetPermissionByRoleId(GetPermissionByIdRequest request)
        {
            var response = _backOfficeRepository.GetPermissionByRoleId(request);
            var permissionId = response.Select(s => s.PermissionId).ToList();
            return new ApiBaseResponse<List<int>>(permissionId);
        }

        public ApiBaseResponse<string> UpdateRolePermission(UpdateRolePermissionRequest request)
        {
            var response = _backOfficeRepository.UpdateRolePermission(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<string> UpdateMemberRole(UpdateMemberRoleRequest request)
        {
            request.CreatedBy = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.UpdateMemberRole(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }


        public ApiBaseResponse<string> UpdateRoleByMemberId(UpdateRoleByMemberIdRequest request)
        {
            request.CreatedBy = _memberDataService.GetCurrentMemberId();
            var response = _backOfficeRepository.UpdateRoleByMemberId(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<List<RoleResponse>> GetRoleByMemberId(GetRoleByMemberIdRequest request)
        {
            var response = _backOfficeRepository.GetRolesBy(request.MemberId);
            return new ApiBaseResponse<List<RoleResponse>>(response);
        }

        public ApiBaseResponse<List<MemberAttendance>> GetAttendanceById(GetAttendanceByIdRequest request)
        {
            request.CheckModel();
            var response = _backOfficeRepository.GetAttendanceById(request);
            var filterReClockAttendance = FilterReClockAttendance(request);
            // var responseData = response.Concat(filterReClockAttendance).ToList();
            return new ApiBaseResponse<List<MemberAttendance>>(response.Concat(filterReClockAttendance).ToList());
        }

        public ApiBaseResponse<List<TakeLeave>> GetLeaveRequestsByMemberId(GetLeaveRequestsByMemberIdRequest request)
        {
            request.CheckModel();
            var response = _backOfficeRepository.GetLeaveRequestsByMemberId(request);
            return new ApiBaseResponse<List<TakeLeave>>(response);
        }

        public ApiBaseResponse<RepositoryBaseResponse> RegisterLeaveAmount(ByMemberIdRequest request)
        {
            request.CheckModel();
            var response = _backOfficeRepository.RegisterLeaveAmount(request);
            response.CheckErrorCode();

            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }

        public ApiBaseResponse<RepositoryBaseResponse> UpdateMemberMonthlyTransferStatus(
            UpdateMemberMonthlyTransferStatusRequest request)
        {
            var response = _backOfficeRepository.UpdateMemberMonthlyTransferStatus(request);
            response.CheckErrorCode();

            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }

        public ApiBaseResponse<RepositoryBaseResponse> BatchUpdateMonthlyTransferStatus(
            BatchUpdateMonthlyTransferStatusRequest request)
        {
            var response = _backOfficeRepository.BatchUpdateMonthlyTransferStatus(request);
            response.CheckErrorCode();
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }

        public ApiBaseResponse<List<PositionResponse>> GetAllPositionByTeamId(GetPositionByTeamIdRequest request)
        {
            var response = _backOfficeRepository.GetAllPositionByTeamId(request);
            return new ApiBaseResponse<List<PositionResponse>>(response);
        }

        public ApiBaseResponse<ResignTransferResponse> GetResignTransfer(GetResignTransferRequest request)
        {
            var response = _backOfficeRepository.GetResignTransfer(request);
            return new ApiBaseResponse<ResignTransferResponse>(response);
        }

        public ApiBaseResponse<string> AddResignTransfer(AddResignTransfer request)
        {
            List<AddMonthlyTransferRequest> requestMonthly = new List<AddMonthlyTransferRequest>();
            if (request.UnpaidLeave > 0)
            {
                requestMonthly.Add(new AddMonthlyTransferRequest()
                {
                    TransferId = 0,
                    TransferTypeId = Convert.ToInt32(TransferTypeEnum.UnpaidLeave),
                    MemberId = request.MemberId,
                    Amount = request.DeductionAmount,
                    DayCount = request.UnpaidLeave,
                    BeneficiaryId = Convert.ToInt32(BeneficiaryTypeEnum.Deduction),
                    Status = 2,
                    TransferDate = DateTime.UtcNow.Date,
                    Remark = "",
                    PayStartDate = request.StartDate,
                    PayEndDate = request.ResignDate,
                    CreateBy = _memberDataService.GetCurrentMemberId(),
                    ModifiedBy = _memberDataService.GetCurrentMemberId(),
                });
            }

            if (request.WorkingDay < 30)
            {
                requestMonthly.Add(new AddMonthlyTransferRequest()
                {
                    TransferId = 0,
                    TransferTypeId = Convert.ToInt32(TransferTypeEnum.NonWorking),
                    MemberId = request.MemberId,
                    Amount = request.DeductionAmount,
                    DayCount = 0,
                    BeneficiaryId = Convert.ToInt32(BeneficiaryTypeEnum.Deduction),
                    Status = 2,
                    TransferDate = DateTime.UtcNow.Date,
                    PayStartDate = request.StartDate,
                    PayEndDate = request.ResignDate,
                    Remark = "",
                    CreateBy = _memberDataService.GetCurrentMemberId(),
                    ModifiedBy = _memberDataService.GetCurrentMemberId(),
                });
            }

            requestMonthly.Add(new AddMonthlyTransferRequest()
            {
                TransferId = 0,
                TransferTypeId = Convert.ToInt32(TransferTypeEnum.MonthlyTransfer),
                MemberId = request.MemberId,
                Amount = request.Salary,
                BeneficiaryId = Convert.ToInt32(BeneficiaryTypeEnum.Unknown),
                Status = 2,
                DayCount = request.WorkingDay,
                TransferDate = DateTime.UtcNow.Date,
                PayStartDate = request.StartDate,
                PayEndDate = request.ResignDate,
                Remark = "",
                CreateBy = _memberDataService.GetCurrentMemberId(),
                ModifiedBy = _memberDataService.GetCurrentMemberId(),
            });
            foreach (var i in requestMonthly)
            {
                var response = _backOfficeRepository.AddMonthlyTransfer(i);
                response.CheckErrorCode();
            }

            return new ApiBaseResponse<string>();
        }

        public ApiBaseResponse<List<DepartmentResponse>> GetAllDepartments()
        {
            var response = _backOfficeRepository.GetAllDepartments();
            return new ApiBaseResponse<List<DepartmentResponse>>(response);
        }
        public ApiBaseResponse<RepositoryBaseResponse> AddDepartment(AddDepartmentRequest request)
        {
            var response = _backOfficeRepository.AddDepartment(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> UpdateDepartment(UpdateDepartmentRequest request)
        {
            var response = _backOfficeRepository.UpdateDepartment(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> DeleteDepartment(int departmentId)
        {
            var response = _backOfficeRepository.DeleteDepartment(departmentId);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        public ApiBaseResponse<RepositoryBaseResponse> AddTeam(AddTeamRequest request)
        {
            var response = _backOfficeRepository.AddTeam(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> UpdateTeam(UpdateTeamRequest request)
        {
            var response = _backOfficeRepository.UpdateTeam(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> DeleteTeam(int teamId)
        {
            var response = _backOfficeRepository.DeleteTeam(teamId);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }

        public ApiBaseResponse<List<PositionResponse>> GetAllPosition()
        {
            var response = _backOfficeRepository.GetAllPosition();
            return new ApiBaseResponse<List<PositionResponse>>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> AddPosition(AddPositionRequest request)
        {
            var response = _backOfficeRepository.AddPosition(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> UpdatePosition(UpdatePositionRequest request)
        {
            var response = _backOfficeRepository.UpdatePosition(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> DeletePosition(int positionId)
        {
            var response = _backOfficeRepository.DeletePosition(positionId);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> AddJobGrade(AddJobGradeRequest request)
        {
            var response = _backOfficeRepository.AddJobGrade(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> UpdateJobGrade(UpdateJobGradeRequest request)
        {
            var response = _backOfficeRepository.UpdateJobGrade(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> DeleteJobGrade(int jobGradeId)
        {
            var response = _backOfficeRepository.DeleteJobGrade(jobGradeId);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> UpdateLocation(UpdateLocationRequest request)
        {
            var response = _backOfficeRepository.UpdateLocation(request);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        
        public ApiBaseResponse<RepositoryBaseResponse> DeleteLocation(int locationId)
        {
            var response = _backOfficeRepository.DeleteLocation(locationId);
            return new ApiBaseResponse<RepositoryBaseResponse>(response);
        }
        #region PrivateMethod

        private TimeSpan? ConvertTime(TimeSpan? serverTime, TimeSpan? userTime)
        {
            Console.WriteLine("server" + serverTime.HasValue);
            if (serverTime.HasValue)
            {
                TimeSpan? serverTimeSpan = serverTime; /* replace with your code to get the server time as a TimeSpan */
                ;

                // Get the user's time zone offset from the request
                TimeSpan?
                    userTimeZoneOffset = userTime; /* replace with your code to get the user request time zone offset */
                ;

                // Create a TimeZoneInfo object for the user's time zone
                TimeZoneInfo userTimeZone = TimeZoneInfo.CreateCustomTimeZone("User Time Zone",
                    userTimeZoneOffset ?? TimeSpan.Zero, "User Time Zone", "User Time Zone");

                // Create a DateTimeOffset object for the server time with the server time zone offset
                DateTimeOffset serverDateTimeOffset =
                    new DateTimeOffset(DateTime.MinValue.Add(serverTimeSpan ?? TimeSpan.Zero), TimeSpan.Zero);
                // Convert the server DateTimeOffset object to the user's time zone
                DateTimeOffset userDateTimeOffset = TimeZoneInfo.ConvertTime(serverDateTimeOffset, userTimeZone);

                // Use the resulting DateTimeOffset object for display or manipulation
                Console.WriteLine($"User local time is: {userDateTimeOffset}");

                return userDateTimeOffset.TimeOfDay;
            }

            return null;
        }

        private IEnumerable<MemberAttendance> FilterReClockAttendance(GetAttendanceByIdRequest request)
        {
            var listMemberAttendance = new List<MemberAttendance> { };
            var responseReClockRecord = _backOfficeRepository.GetAllReClockRecordsById(request);
            var groupedResult = responseReClockRecord.GroupBy(x => x.Date);

            foreach (var i in groupedResult)
            {
                var memberAttendance = new MemberAttendance();
                foreach (var r in i)
                {
                    memberAttendance.MemberId = r.MemberId;
                    memberAttendance.Username = r.Username;
                    memberAttendance.TeamName = r.TeamName;
                    memberAttendance.WorkDate = r.Date;
                    memberAttendance.ReClockIn = r.IsClockIn ? r.Time : memberAttendance.ReClockIn;
                    memberAttendance.ReClockOut = r.IsClockIn == false ? r.Time : memberAttendance.ReClockOut;
                    memberAttendance.ReClockInLocation = r.IsClockIn ? r.Location : memberAttendance.ReClockInLocation;
                    memberAttendance.ReClockOutLocation =
                        r.IsClockIn == false ? r.Location : memberAttendance.ReClockOutLocation;
                    memberAttendance.ClockInRemark = r.IsClockIn ? r.Reason : memberAttendance.ClockInRemark;
                    memberAttendance.ClockOutRemark = r.IsClockIn == false ? r.Reason : memberAttendance.ClockOutRemark;
                }

                listMemberAttendance.Add(memberAttendance);
            }

            return listMemberAttendance;
        }

        private DataTable CreateDataTable(List<DateTime> workingDays)
        {
            var tvpAbsentee = new DataTable();
            tvpAbsentee.Columns.Add("WorkDate", typeof(DateTime));
            foreach (var workingDay in workingDays)
            {
                var row = tvpAbsentee.NewRow();
                row["WorkDate"] = workingDay;
                tvpAbsentee.Rows.Add(row);
            }

            return tvpAbsentee;
        }

        private void LeaveApprovalNotificationById(RequestApprovalRequest request)
        {
            var profile = _memberRepository.GetProfile(request.MemberId);

            var content = ((Func<int, string>) (a =>
            {
                switch (a)
                {
                    case 1:
                        return "Approved";
                    case 2:
                        return "Not Approved";
                    default:
                        return "Pending";
                }
            }))(request.IsApproved);
            var notificationRequest = new SendNotificationByExternalIdRequest
            {
                ExternalIds = new List<string> {profile.Email}
            };

            var notificationMessage = new NotificationMessage()
            {
                Title = "Leave Request Approval",
                Content = content,
            };
            notificationRequest.Messages = new List<NotificationMessage> {notificationMessage};
            _notificationService.SendNotificationByExternalIds(notificationRequest);
        }
        
        #endregion PrivateMethod
    }
}