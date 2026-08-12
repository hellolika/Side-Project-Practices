using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using System.Data;

namespace HRMS.Repositories.Interfaces
{
    public interface IBackOfficeRepository
    {
        IEnumerable<GetAbsenteeResponse> GetAbsentee(DataTable tvpWorkingDay);

        List<GetMembersResponse> GetMembers();

        RepositoryBaseResponse LeaveRequestApproval(RequestApprovalRequest request);

        RepositoryBaseResponse ReClockRequestApproval(RequestApprovalRequest request);

        RepositoryBaseResponse AddNewLocation(AddLocationRequest location);

        RegisterMemberResponse RegisterMember(RegisterMemberRequest request);

        RepositoryBaseResponse EditMember(EditMemberInfoRequest request);

        RepositoryBaseResponse DeleteMember(DeleteMemberRequest request);

        List<LeavesResponse> GetAllLeaves(TimeRangeRequest request);

        List<ReClocksResponse> GetAllReClock(TimeRangeRequest request);

        RepositoryBaseResponse ResetPassword(PasswordBaseRequest request);

        List<MemberLeaveAmount> GetAllLeaveAmount();

        RepositoryBaseResponse UpdateLeaveAmount(UpdateLeaveAmountRequest request);

        RepositoryBaseResponse PopulateDefaultLeaves();
        RepositoryBaseResponse PopulateMemberRoleRecord(PopulateMemberRoleRecordRequest request);
        RepositoryBaseResponse AddAnnouncement(AddAnnouncementRequest request);

        RepositoryBaseResponse EditAnnouncement(EditAnnouncementRequest request);

        RepositoryBaseResponse DeleteAnnouncement(int id);
        
        RepositoryBaseResponse AddTransferType(AddTransferTypeRequest request);
        
        List<TransferTypeResponse> GetAllTransferTypes();

        List<TransferResponse> GenerateMemberMonthlyTransfer(GenerateMemberMonthlyTransferRequest request);

        RepositoryBaseResponse AddMonthlyTransfer(AddMonthlyTransferRequest request);
        
        RepositoryBaseResponse DeleteMonthlyTransfer(DeleteMonthlyTransferRequest request);

        List<TransferResponse> GetMonthlyTransfers(GenerateMemberMonthlyTransferRequest request);
        
        RepositoryBaseResponse AddOrEditLeaveRecord(AddOrEditLeaveRecordRequest request);

        RepositoryBaseResponse AddUnpaidLeave(int memberId);

        RepositoryBaseResponse EditMemberLeaves(EditMemberLeaveRequest request);

        List<LeaveAmountResponse> GetLeaveAmountByMemberId(GetLeaveAmountByMemberIdRequest request);

        List<MemberAttendance> GetAllAttendance(GetAllAttendanceRequest request);

        List<RoleResponse> GetAllRole();
        
        RepositoryBaseResponse AddMemberRole(AddMemberRoleRequest request); 
        
        RepositoryBaseResponse DeleteMemberRole(DeleteMemberRoleRequest request);

        List<GetAllPermissionResponse> GetAllPermission();

        DashboardResponse GetDashboard();

        RepositoryBaseResponse AddRole(AddRoleRequest request);

        List<GetPermissionByIdRepsonse> GetPermissionByRoleId(GetPermissionByIdRequest request);

        RepositoryBaseResponse UpdateRolePermission(UpdateRolePermissionRequest request);

        RepositoryBaseResponse UpdateMemberRole(UpdateMemberRoleRequest request);

        List<MemberRoleReponse> GetMemberByRoleId(int roleId);

        RepositoryBaseResponse UpdateRoleByMemberId(UpdateRoleByMemberIdRequest request);

        List<RoleResponse> GetRolesBy(int memberId);

        List<BeneficiaryList> GetBeneficiaryType();

        List<MemberAttendance> GetAttendanceById(GetAttendanceByIdRequest request);

        List<TakeLeave> GetLeaveRequestsByMemberId(GetLeaveRequestsByMemberIdRequest request);

        List<ReClockRecord> GetAllReClockRecordsById(GetAttendanceByIdRequest request);

        RepositoryBaseResponse RegisterLeaveAmount(ByMemberIdRequest request);

        RepositoryBaseResponse AddLeaveAmountToAllMember(int amount = 1);
        
        RepositoryBaseResponse UpdateMemberMonthlyTransferStatus(UpdateMemberMonthlyTransferStatusRequest request);

        RepositoryBaseResponse BatchUpdateMonthlyTransferStatus(BatchUpdateMonthlyTransferStatusRequest request);

        List<PositionResponse> GetAllPositionByTeamId(GetPositionByTeamIdRequest request);

        ResignTransferResponse GetResignTransfer(GetResignTransferRequest request);
        
        List<DepartmentResponse> GetAllDepartments();
        RepositoryBaseResponse UpdateDepartment(UpdateDepartmentRequest request);

        RepositoryBaseResponse AddDepartment(AddDepartmentRequest request);
        RepositoryBaseResponse DeleteDepartment(int departmentId);
        
        RepositoryBaseResponse AddTeam(AddTeamRequest request);
        
        RepositoryBaseResponse UpdateTeam(UpdateTeamRequest request);
        
        RepositoryBaseResponse DeleteTeam(int teamId);
        
        RepositoryBaseResponse AddPosition(AddPositionRequest request);
        
        RepositoryBaseResponse UpdatePosition(UpdatePositionRequest request);
        
        RepositoryBaseResponse DeletePosition(int positionId);
        
        RepositoryBaseResponse AddJobGrade(AddJobGradeRequest request);
        
        RepositoryBaseResponse UpdateJobGrade(UpdateJobGradeRequest request);
        
        RepositoryBaseResponse DeleteJobGrade(int jobGradeId);
        
        RepositoryBaseResponse UpdateLocation(UpdateLocationRequest request);
        
        RepositoryBaseResponse DeleteLocation(int locationId);
        
        List<PositionResponse> GetAllPosition();
        
        List<JobGradeResponse> GetAllJobGrade();
    }
}