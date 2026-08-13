using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;

namespace HRMS.Services.Interfaces
{
    public interface IMemberService
    {
        ApiBaseResponse<string> CancelLeave(CancelLeaveRequest request);

        ApiBaseResponse<string> ChangePassword(ChangePasswordRequest request);

        ApiBaseResponse<ClockStatusResponse> CheckClockStatus(ClockStatusRequest request);

        ApiBaseResponse<string> DoClock(DoClockRequest request);

        ApiBaseResponse<SubmitFormBaseResponse> DoReclock(DoReClockRequest request);

        ApiBaseResponse<AllRequestFormsResponse> GetAllRequestForms(BaseRequest request);

        ApiBaseResponse<List<LeaveAmountResponse>> GetMemberLeaveAmount();

        ApiBaseResponse<List<LeaveAmountResponseV2>> GetMemberLeaveAmountV2(int memberId = 0);

        ApiBaseResponse<List<LeaveType>> GetAllLeaveType();

        ApiBaseResponse<List<LocationDetails>> GetLocation();

        ApiBaseResponse<GetAnnouncementResponse> GetAnnouncements(GetAnnouncementRequest request);

        ApiBaseResponse<MemberProfile> GetProfile(int memberId = 0, bool isAdmin = true, int jwtMemberId = 0);

        ApiBaseResponse<int> UpdateProfile(UpdateProfileRequest request);

        ApiBaseResponse<List<TeamInfo>> GetAllTeamInfo();

        ApiBaseResponse<List<TimeTableResponse>> GetTimeTable(TimeTableRequest request);

        ApiBaseResponse<SubmitFormBaseResponse> TakeLeave(TakeLeaveRequest request);
        
        ApiBaseResponse<List<TransferResponse>> GetTransfers();
        Task<ApiBaseResponse<UploadImageResponse>> UploadLeaveImage(UploadImageRequest request);
        
        ApiBaseResponse<GetAvailableSettingForMemberOperationResponse> GetAvailableSettingForMemberOperation();
        
        ApiBaseResponse<RepositoryBaseResponse> UpsertSlackUsers(List<SlackUser> slackUsers);
    }
}