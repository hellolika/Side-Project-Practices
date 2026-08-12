using System.Data;
using DBCiper.Helper;
using DBCiper.Repositories;
using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Models.Settings;
using HRMS.Repositories.Interfaces;
using Microsoft.Extensions.Options;

namespace HRMS.Repositories
{
    public class MemberRepository : BaseRepository, IMemberRepository
    {
        private MemberStoreProcedureSettings _memberStoreProcedureSettings;

        public MemberRepository(IConfiguration configuration,
            IOptionsMonitor<MemberStoreProcedureSettings> storeProcedureSettings)
            : base(configuration)
        {
            _memberStoreProcedureSettings = storeProcedureSettings.CurrentValue;
            storeProcedureSettings.OnChange(newValue => { _memberStoreProcedureSettings = newValue; });
        }

        public LoginResponse Login(string email, string password)
        {
            return Query<LoginResponse>(_memberStoreProcedureSettings.Login,
                new
                {
                    email, password
                }).First();
        }

        public RepositoryBaseResponse DoClock(DoClockRequest request)
        {
            return Query<RepositoryBaseResponse>(_memberStoreProcedureSettings.DoClock,
                new
                {
                    memberId = request.MemberId,
                    date = request.Date.Date,
                    offSetInMinutes = request.TimeZone.Value.TotalMinutes - DateTimeOffset.Now.Offset.TotalMinutes,
                    isClockIn = request.IsClockIn,
                    isInCompany = request.IsInCompany,
                    location = request.Location,
                    clockInRemark = request.ClockInRemark,
                    clockOutRemark = request.ClockOutRemark
                }).First();
        }

        public ClockStatusResponse CheckClockStatus(ClockStatusRequest request)
        {
            return Query<ClockStatusResponse>(_memberStoreProcedureSettings.CheckClockStatus,
                new
                {
                    memberId = request.MemberId,
                    date = request.Date.Value.Date
                }).First();
        }

        public SubmitFormBaseResponse DoReclock(DoReClockRequest request)
        {
            return Query<SubmitFormBaseResponse>(_memberStoreProcedureSettings.DoReclock,
                new
                {
                    memberId = request.MemberId,
                    date = request.Date.Value.Date,
                    time = request.Time.Value,
                    reason = request.Reason,
                    location = request.Location,
                    isClockIn = request.IsClockIn,
                }).First();
        }

        public SubmitFormBaseResponse TakeLeave(TakeLeaveRequest request)
        {
            return Query<SubmitFormBaseResponse>(_memberStoreProcedureSettings.TakeLeave,
                new
                {
                    memberId = request.MemberId,
                    numberOfDay = request.NumberOfDay,
                    startDate = request.StartDate.DateTime,
                    endDate = request.EndDate.DateTime,
                    leaveType = request.LeaveType,
                    image = request.ImagePath,
                    reason = request.Reason,
                }).First();
        }

        public SubmitFormBaseResponse UpdateTakeLeave(TakeLeaveRequest request)
        {
            return Query<SubmitFormBaseResponse>(_memberStoreProcedureSettings.UpdateTakeLeave,
                new
                {
                    requestId = request.RequestId,
                    memberId = request.MemberId,
                    numberOfDay = request.NumberOfDay,
                    startDate = request.StartDate.DateTime,
                    endDate = request.EndDate.DateTime,
                    leaveType = request.LeaveType,
                    image = request.ImagePath,
                    reason = request.Reason,
                }).First();
        }

        public List<MemberPermissionResponse> GetPermissionsByMemberId(int memberId)
        {
            return Query<MemberPermissionResponse>(_memberStoreProcedureSettings.GetMemberPermission,
                new
                {
                    memberId
                }).ToList();
        }

        public List<LeaveAmountResponse> GetMemberLeaveAmount(int memberId)
        {
            return Query<LeaveAmountResponse>(_memberStoreProcedureSettings.GetMemberLeaveAmount, new {memberId})
                .ToList();
        }

        public List<LeaveAmountResponseV2> GetMemberLeaveAmountV2(int memberId)
        {
            return Query<LeaveAmountResponseV2>(_memberStoreProcedureSettings.GetMemberLeaveAmountV2, new {memberId})
                .ToList();
        }

        public List<TakeLeave> GetMemberLeaveRequestRecords(int memberId)
        {
            return Query<TakeLeave>(_memberStoreProcedureSettings.GetLeaveRecordsByMemberId,
                new
                {
                    memberId
                }).ToList();
        }

        public List<ReClock> GetAllRequestFormsReClock(int memberId)
        {
            return Query<ReClock>(_memberStoreProcedureSettings.GetAllReClockRecords,
                new
                {
                    memberId
                }).ToList();
        }

        public MemberProfile GetProfile(int memberId)
        {
            return Query<MemberProfile>(_memberStoreProcedureSettings.GetProfile,
                new
                {
                    memberId
                }).FirstOrDefault();
        }

        public RepositoryBaseResponse UpdateProfile(UpdateProfileRequest request, int memberId)
        {
            return Query<RepositoryBaseResponse>(_memberStoreProcedureSettings.UpdateProfile,
                new
                {
                    memberId = memberId,
                    teamId = request.TeamId,
                    workLocationId = request.WorkLocationId,
                    email = request.Email,
                    phoneNumber = request.PhoneNumber,
                    address = request.Address,
                    bankAccount = request.BankAccount,
                    remark = request.Remark,
                    bankName = request.BankName,
                    position = request.Position
                }).First();
        }

        public RepositoryBaseResponse CancelLeave(CancelLeaveRequest request)
        {
            return Query<RepositoryBaseResponse>(_memberStoreProcedureSettings.CancelLeave,
                new
                {
                    memberId = request.MemberId,
                    requestId = request.RequestId,
                }).First();
        }

        public List<LeaveType> GetAllLeaveType()
        {
            return Query<LeaveType>(_memberStoreProcedureSettings.GetAllLeaveType).ToList();
        }

        public List<TeamInfo> GetAllTeamInfo()
        {
            return Query<TeamInfo>(_memberStoreProcedureSettings.GetAllTeamInfo).ToList();
        }

        public List<LocationDetails> GetLocation()
        {
            return Query<LocationDetails>(_memberStoreProcedureSettings.GetLocation).ToList();
        }

        public RepositoryBaseResponse ChangePassword(ChangePasswordRequest request)
        {
            return Query<RepositoryBaseResponse>(_memberStoreProcedureSettings.ChangePassword,
                new
                {
                    memberId = request.MemberId,
                    oldPassword = request.Password,
                    newPassword = request.NewPassword
                }).First();
        }

        public bool IsMemberExist(int id)
        {
            return Query<bool>(_memberStoreProcedureSettings.IsMemberExist,
                new
                {
                    id
                }).FirstOrDefault();
        }

        public List<Attendance> GetMemberAttendances(MemberTimeTableRequest request)
        {
            return Query<Attendance>(_memberStoreProcedureSettings.GetSingleMemberAttendancesByDateRange,
                new
                {
                    memberId = request.MemberId,
                    startDate = request.StartDate.Date,
                    endDate = request.EndDate.Date
                }).ToList();
        }

        public List<ReClockRecord> GetMemberReClockRecords(MemberTimeTableRequest request)
        {
            return Query<ReClockRecord>(_memberStoreProcedureSettings.GetSingleMemberReclockRecordsByDateRange,
                new
                {
                    memberId = request.MemberId,
                    startDate = request.StartDate.Date,
                    endDate = request.EndDate.Date
                }).ToList();
        }

        public List<TakeLeaveRecord> GetMemberTakeLeaveRecords(MemberTimeTableRequest request)
        {
            return Query<TakeLeaveRecord>(_memberStoreProcedureSettings.GetMemberLeaveRequestByDates,
                new
                {
                    memberId = request.MemberId,
                    startDate = request.StartDate.Date,
                    endDate = request.EndDate.Date
                }).ToList();
        }

        public List<GetAnnouncementResponseItem> GetAnnouncements(GetAnnouncementRequest request)
        {
            return Query<GetAnnouncementResponseItem>(_memberStoreProcedureSettings.GetAnnouncements,
                new
                {
                    request.Page,
                    request.ItemPerPage
                }).ToList();
        }

        public int GetTotalAnnouncements()
        {
            return Query<int>(_memberStoreProcedureSettings.GetTotalAnnouncements).First();
        }

        public List<TransferResponse> GetTransfers(int memberId)
        {
            return Query<TransferResponse>(_memberStoreProcedureSettings.GetMemberTransfers, new {memberId}).ToList();
        }
        
        public List<GetDepartmentManagerResponse> GetDepartmentManager(int memberId)
        {
            return Query<GetDepartmentManagerResponse>(_memberStoreProcedureSettings.GetDepartmentManager, new {memberId}).ToList();
        }
        
        public RepositoryBaseResponse UpsertSlackUsers(DataTable dataTable)
        {
            var response =  Query<RepositoryBaseResponse>(_memberStoreProcedureSettings.UpsertSlackUsers, new
            {
                slackUserData = dataTable
            }).First();
            return response;
        }
        
    }
}