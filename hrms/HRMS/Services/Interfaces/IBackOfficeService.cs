using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;

namespace HRMS.Services.Interfaces
{
    public interface IBackOfficeService
    {
        ApiBaseResponse<List<GetAbsenteeResponse>> GetAllAbsentee(GetAbsenteeRequest request);

        ApiBaseResponse<List<GetMembersResponse>> GetMembers(JwtData jwtData = null);

        ApiBaseResponse<string> RegisterMember(RegisterMemberRequest request);

        ApiBaseResponse<string> EditMember(EditMemberInfoRequest request);

        ApiBaseResponse<string> DeleteMember(DeleteMemberRequest request);

        ApiBaseResponse<List<LeavesResponse>> GetAllLeaveRequests(TimeRangeRequest request);

        ApiBaseResponse<string> AddNewLocation(AddLocationRequest location);

        ApiBaseResponse<string> LeaveRequestApproval(RequestApprovalRequest request);

        ApiBaseResponse<List<ReClocksResponse>> GetAllReClockRequests(TimeRangeRequest request);

        ApiBaseResponse<string> ReClockRequestApproval(RequestApprovalRequest request);

        ApiBaseResponse<string> ResetPassword(PasswordBaseRequest request);

        ApiBaseResponse<List<GetAllLeaveAmountResponse>> GetAllLeaveAmount();

        ApiBaseResponse<string> UpdateLeaveAmount(UpdateLeaveAmountRequest request);

        ApiBaseResponse<string> PopulateDefaultLeaves();

        ApiBaseResponse<string> PopulateMemberRoleRecord(PopulateMemberRoleRecordRequest roleId);

        ApiBaseResponse<string> AddAnnouncement(AddAnnouncementRequest request);

        ApiBaseResponse<string> EditAnnouncement(EditAnnouncementRequest request);

        ApiBaseResponse<string> DeleteAnnouncement(int id);

        ApiBaseResponse<string> AddTransferType(AddTransferTypeRequest request);

        ApiBaseResponse<List<TransferTypeResponse>> GetAllTransferTypes();

        ApiBaseResponse<List<TransferResponse>> GenerateMemberMonthlyTransfer(GenerateMemberMonthlyTransferRequest request);

        ApiBaseResponse<string> AddMonthlyTransfer(AddMonthlyTransferRequest request);
        
        ApiBaseResponse<string> DeleteMonthlyTransfer(DeleteMonthlyTransferRequest request);
        
        ApiBaseResponse<GetMontlyTransferResponse> GetMonthlyTransfers(GenerateMemberMonthlyTransferRequest request, bool isAdmin);
        
        ApiBaseResponse<SubmitFormBaseResponse> AddOrEditLeaveRecord(AddOrEditLeaveRecordRequest request);

        ApiBaseResponse<string> EditMemberLeaves(EditMemberLeaveRequest request);   

        ApiBaseResponse<List<LeaveAmountResponse>> GetLeaveAmountByMemberId(GetLeaveAmountByMemberIdRequest request);

        ApiBaseResponse<List<MemberAttendance>> GetAllAttendance(GetAllAttendanceRequest request);

        ApiBaseResponse<DashboardResponse> GetDashboard();
        
        ApiBaseResponse<List<RoleResponse>> GetAllRole();

        ApiBaseResponse<string> AddMemberRole(AddMemberRoleRequest request);
        
        ApiBaseResponse<string> DeleteMemberRole(DeleteMemberRoleRequest request);
        
        ApiBaseResponse<Dictionary<string, List<GetAllPermissionResponse>>> GetAllPermission();
        
        ApiBaseResponse<string> AddRole(AddRoleRequest request);

        ApiBaseResponse<List<int>> GetPermissionByRoleId(GetPermissionByIdRequest request);

        ApiBaseResponse<string> UpdateRolePermission(UpdateRolePermissionRequest request);

        ApiBaseResponse<string> UpdateMemberRole(UpdateMemberRoleRequest request);

        ApiBaseResponse<string> UpdateRoleByMemberId(UpdateRoleByMemberIdRequest request);

        ApiBaseResponse<List<RoleResponse>> GetRoleByMemberId(GetRoleByMemberIdRequest request);

        ApiBaseResponse<List<MemberAttendance>> GetAttendanceById(GetAttendanceByIdRequest request);

        ApiBaseResponse<List<TakeLeave>> GetLeaveRequestsByMemberId(GetLeaveRequestsByMemberIdRequest request);

        ApiBaseResponse<RepositoryBaseResponse> RegisterLeaveAmount(ByMemberIdRequest request);
        
        ApiBaseResponse<RepositoryBaseResponse> UpdateMemberMonthlyTransferStatus(UpdateMemberMonthlyTransferStatusRequest request);
        ApiBaseResponse<RepositoryBaseResponse> BatchUpdateMonthlyTransferStatus(BatchUpdateMonthlyTransferStatusRequest request);

        ApiBaseResponse<List<PositionResponse>> GetAllPositionByTeamId(GetPositionByTeamIdRequest request);
        
        ApiBaseResponse<ResignTransferResponse> GetResignTransfer(GetResignTransferRequest request);
        
        ApiBaseResponse<string> AddResignTransfer(AddResignTransfer request);
        
        ApiBaseResponse<List<DepartmentResponse>> GetAllDepartments();
        
        ApiBaseResponse<RepositoryBaseResponse> AddDepartment(AddDepartmentRequest request);

        ApiBaseResponse<RepositoryBaseResponse> DeleteDepartment(int departmentId);

        ApiBaseResponse<RepositoryBaseResponse> UpdateDepartment(UpdateDepartmentRequest request);
        
        ApiBaseResponse<RepositoryBaseResponse> AddTeam(AddTeamRequest request);
        
        ApiBaseResponse<RepositoryBaseResponse> UpdateTeam(UpdateTeamRequest request);
        
        ApiBaseResponse<RepositoryBaseResponse> DeleteTeam(int teamId);
        ApiBaseResponse<List<PositionResponse>> GetAllPosition();
        ApiBaseResponse<RepositoryBaseResponse> AddPosition(AddPositionRequest request);
        
        ApiBaseResponse<RepositoryBaseResponse> UpdatePosition(UpdatePositionRequest request);
        
        ApiBaseResponse<RepositoryBaseResponse> DeletePosition(int positionId);

        ApiBaseResponse<RepositoryBaseResponse> AddJobGrade(AddJobGradeRequest request);

        ApiBaseResponse<RepositoryBaseResponse> UpdateJobGrade(UpdateJobGradeRequest request);
        
        ApiBaseResponse<RepositoryBaseResponse> DeleteJobGrade(int jobGradeId);
        
        ApiBaseResponse<RepositoryBaseResponse> UpdateLocation(UpdateLocationRequest request);
        
        ApiBaseResponse<RepositoryBaseResponse> DeleteLocation(int locationId);
    }
}