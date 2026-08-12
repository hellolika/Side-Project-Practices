using System.Data;
using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;

namespace HRMS.Repositories.Interfaces
{
    public interface IMemberRepository
    {
        RepositoryBaseResponse CancelLeave(CancelLeaveRequest request);

        RepositoryBaseResponse ChangePassword(ChangePasswordRequest request);

        ClockStatusResponse CheckClockStatus(ClockStatusRequest request);

        RepositoryBaseResponse DoClock(DoClockRequest request);

        SubmitFormBaseResponse DoReclock(DoReClockRequest request);

        List<ReClock> GetAllRequestFormsReClock(int memberId);

        List<TakeLeave> GetMemberLeaveRequestRecords(int memberId);

        List<LeaveAmountResponse> GetMemberLeaveAmount(int memberId);

        List<LeaveAmountResponseV2> GetMemberLeaveAmountV2(int memberId);

        List<LeaveType> GetAllLeaveType();

        List<LocationDetails> GetLocation();

        MemberProfile GetProfile(int Id);

        RepositoryBaseResponse UpdateProfile(UpdateProfileRequest request, int memberId);

        List<TeamInfo> GetAllTeamInfo();

        bool IsMemberExist(int id);

        LoginResponse Login(string email, string password);

        SubmitFormBaseResponse TakeLeave(TakeLeaveRequest request);

        SubmitFormBaseResponse UpdateTakeLeave(TakeLeaveRequest request);

        List<Attendance> GetMemberAttendances(MemberTimeTableRequest request);

        List<ReClockRecord> GetMemberReClockRecords(MemberTimeTableRequest request);

        List<TakeLeaveRecord> GetMemberTakeLeaveRecords(MemberTimeTableRequest request);

        List<GetAnnouncementResponseItem> GetAnnouncements(GetAnnouncementRequest request);

        int GetTotalAnnouncements();
        
        List<TransferResponse> GetTransfers(int memberId);
        
        List<MemberPermissionResponse> GetPermissionsByMemberId(int memberId);

        List<GetDepartmentManagerResponse> GetDepartmentManager(int memberId);
        RepositoryBaseResponse UpsertSlackUsers(DataTable dataTable);
    }
}