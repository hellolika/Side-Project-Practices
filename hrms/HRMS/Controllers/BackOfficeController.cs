using HRMS.Enum;
using HRMS.Filters;
using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Services.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace HRMS.Controllers
{
    [Route("/api/[controller]")]
    [ServiceFilter(typeof(AuthenticateJwt))]
    [ApiController]
    public class BackOfficeController : ControllerBase
    {
        private readonly IBackOfficeService _backOfficeService;

        public BackOfficeController(IBackOfficeService backOfficeService)
        {
            _backOfficeService = backOfficeService;
        }
        
        [HasPermission(PermissionEnum.CanGetDashboard)]
        [HttpGet("GetDashboard")]
        public ApiBaseResponse<DashboardResponse> GetDashboard()
        {
            return _backOfficeService.GetDashboard();
        }

        
        [HasPermission(PermissionEnum.CanGetMembers)]
        [HttpGet("GetMembers")]
        public ApiBaseResponse<List<GetMembersResponse>> GetMembers()
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            // var canSeeMemberSalary = jwtData.Permissions.Contains(Convert.ToInt32(PermissionEnum.CanSeeMemberSalary).ToString());
            return _backOfficeService.GetMembers(jwtData);
        }
        
        [HasPermission(PermissionEnum.CanGetAllLeaveRequests)]
        [HttpPost("GetAllLeaveRequests")]
        public ApiBaseResponse<List<LeavesResponse>> GetAllLeaveRequests(TimeRangeRequest request)
        {
            return _backOfficeService.GetAllLeaveRequests(request);
        }
        
        [HasPermission(PermissionEnum.CanGetAllReClockRequests)]
        [HttpPost("GetAllReclockRequests")]
        public ApiBaseResponse<List<ReClocksResponse>> GetAllReClockRequests(TimeRangeRequest request)
        {
            return _backOfficeService.GetAllReClockRequests(request);
        }
        
        [HasPermission(PermissionEnum.CanGetAllAbsentee)]
        [HttpPost("GetAllAbsentee")]
        public ApiBaseResponse<List<GetAbsenteeResponse>> GetAllAbsentee(GetAbsenteeRequest request)
        {
            return _backOfficeService.GetAllAbsentee(request);
        }

        [HasPermission(PermissionEnum.CanLeaveRequestApproval)]
        [HttpPost("LeaveRequestApproval")]
        public ApiBaseResponse<string> LeaveRequestApproval(RequestApprovalRequest request)
        {
            return _backOfficeService.LeaveRequestApproval(request);
        }

        [HasPermission(PermissionEnum.CanReClockRequestApproval)]
        [HttpPost("ReClockRequestApproval")]
        public ApiBaseResponse<string> ReClockRequestApproval(RequestApprovalRequest request)
        {
            return _backOfficeService.ReClockRequestApproval(request);
        }
        
        [HasPermission(PermissionEnum.CanAddNewLocation)]
        [HttpPost("AddNewLocation")]
        public ApiBaseResponse<string> AddNewLocation(AddLocationRequest location)
        {
            return _backOfficeService.AddNewLocation(location);
        }
        
        [HasPermission(PermissionEnum.CanUpdateLocation)]
        [HttpPost("UpdateLocation")]
        public ApiBaseResponse<RepositoryBaseResponse> UpdateLocation(UpdateLocationRequest location)
        {
            return _backOfficeService.UpdateLocation(location);
        }
        
        
        [HasPermission(PermissionEnum.CanDeleteLocation)]
        [HttpDelete("DeleteLocation")]
        public ApiBaseResponse<RepositoryBaseResponse> DeleteLocation(int locationId)
        {
            return _backOfficeService.DeleteLocation(locationId);
        }
        
        
        [HasPermission(PermissionEnum.CanRegisterMember)]
        [HttpPost("RegisterMember")]
        public ApiBaseResponse<string> RegisterMember(RegisterMemberRequest request)
        {
            return _backOfficeService.RegisterMember(request);
        }
        
        [HasPermission(PermissionEnum.CanEditMember)]
        [HttpPost("EditMember")]
        public ApiBaseResponse<string> EditMember(EditMemberInfoRequest request)
        {
            return _backOfficeService.EditMember(request);
        }
        
        [HasPermission(PermissionEnum.CanDeleteMember)]
        [HttpDelete("DeleteMember")]
        public ApiBaseResponse<string> DeleteMember(DeleteMemberRequest request)
        {
            return _backOfficeService.DeleteMember(request);
        }
        
       [HasPermission(PermissionEnum.CanResetMemberPassword)]
        [HttpPost("ResetMemberPassword")]
        public ApiBaseResponse<string> ResetPassword(PasswordBaseRequest request)
        {
            return _backOfficeService.ResetPassword(request);
        }
        
        [HasPermission(PermissionEnum.CanGetAllLeaveAmount)]
        [HttpGet("GetAllLeaveAmount")]
        public ApiBaseResponse<List<GetAllLeaveAmountResponse>> GetAllLeaveAmount()
        {
            return _backOfficeService.GetAllLeaveAmount();
        }
        
        [HasPermission(PermissionEnum.CanUpdateLeaveAmount)]
        [HttpPost("UpdateLeaveAmount")]
        public ApiBaseResponse<string> UpdateLeaveAmount(UpdateLeaveAmountRequest request)
        {
            return _backOfficeService.UpdateLeaveAmount(request);
        }

        [HasPermission(PermissionEnum.CanPopulateDefaultLeaves)]
        [HttpGet("PopulateDefaultLeaves")]
        public ApiBaseResponse<string> PopulateDefaultLeaves()
        {
            return _backOfficeService.PopulateDefaultLeaves();
        }
        
        [HasPermission(PermissionEnum.CanPopulateMemberRoleRecord)]
        [HttpPost("PopulateMemberRoleRecord")]
        public ApiBaseResponse<string> PopulateMemberRoleRecord(PopulateMemberRoleRecordRequest request)
        {
            return _backOfficeService.PopulateMemberRoleRecord(request);
        }
        
        [HasPermission(PermissionEnum.CanAddAnnouncement)]
        [HttpPost("AddAnnouncement")]
        public ApiBaseResponse<string> AddAnnouncement(AddAnnouncementRequest request)
        {
            return _backOfficeService.AddAnnouncement(request);
        }
        
        [HasPermission(PermissionEnum.CanEditAnnouncement)]
        [HttpPut("EditAnnouncement")]
        public ApiBaseResponse<string> EditAnnouncement(EditAnnouncementRequest request)
        {
            return _backOfficeService.EditAnnouncement(request);
        }
        
        [HasPermission(PermissionEnum.CanDeleteAnnouncement)]
        [HttpDelete("DeleteAnnouncement")]
        public ApiBaseResponse<string> DeleteAnnouncement(int id)
        {
            return _backOfficeService.DeleteAnnouncement(id);
        }
        
        [HasPermission(PermissionEnum.CanGetAllTransferTypes)]
        [HttpGet("GetAllTransferTypes")]
        public ApiBaseResponse<List<TransferTypeResponse>> GetAllTransferTypes()
        {
            return _backOfficeService.GetAllTransferTypes();
        }
        
        [HasPermission(PermissionEnum.CanAddTransferType)]
        [HttpPost("AddTransferType")]
        public ApiBaseResponse<string> AddTransferType(AddTransferTypeRequest request)
        {
            return _backOfficeService.AddTransferType(request);
            
        }
        
        [HasPermission(PermissionEnum.CanGenerateMemberMonthlyTransfer)]
        [HttpPost("GenerateMemberMonthlyTransfer")]
        public ApiBaseResponse<List<TransferResponse>> GenerateMemberMonthlyTransfer(
            GenerateMemberMonthlyTransferRequest request)
        {
            return _backOfficeService.GenerateMemberMonthlyTransfer(request);
        }
        
        [HasPermission(PermissionEnum.CanAddMonthlyTransfer)]
        [HttpPost("AddMonthlyTransfer")]
        public ApiBaseResponse<string> AddMonthlyTransfer(AddMonthlyTransferRequest request)
        {
            return _backOfficeService.AddMonthlyTransfer(request);
        }
        
       [HasPermission(PermissionEnum.CanDeleteMonthlyTransfer)]
        [HttpDelete("DeleteMonthlyTransfer")]
        public ApiBaseResponse<string> DeleteMonthlyTransfer(DeleteMonthlyTransferRequest request)
        {
            return _backOfficeService.DeleteMonthlyTransfer(request);
        }
        
        [HasPermission(PermissionEnum.CanGetMonthlyTransfer)]
        [HttpPost("GetMonthlyTransfers")]
        public ApiBaseResponse<GetMontlyTransferResponse> GetMonthlyTransfers(
            GenerateMemberMonthlyTransferRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            return  _backOfficeService.GetMonthlyTransfers(request, jwtData.IsAdmin());
        }
        
        [HasPermission(PermissionEnum.CanAddLeaveRecord,optionalClaim: PermissionEnum.CanEditLeaveReacord)]
        [HttpPost("AddOrEditLeaveRecord")]
        public ApiBaseResponse<SubmitFormBaseResponse> AddOrEditLeaveRecord(
            AddOrEditLeaveRecordRequest request)
        {
            return _backOfficeService.AddOrEditLeaveRecord(request);
        }

        // [ServiceFilter(typeof(AdminPermissionFilter))]
        [HasPermission(PermissionEnum.CanEditMemberLeave)]
        [HttpPost("EditMemberLeaves")]
        public ApiBaseResponse<string> EditMemberLeaves(
            EditMemberLeaveRequest request)
        {
            return _backOfficeService.EditMemberLeaves(request);
        }
        
        [HasPermission(PermissionEnum.CanGetLeaveAmountByMemberId)]
        // [ServiceFilter(typeof(AdminPermissionFilter))]
        [HttpPost("GetLeaveAmountByMemberId")]
        public ApiBaseResponse<List<LeaveAmountResponse>> GetLeaveAmountByMemberId(
            GetLeaveAmountByMemberIdRequest request)
        {
            return _backOfficeService.GetLeaveAmountByMemberId(request);
        }

        // [ServiceFilter(typeof(AdminPermissionFilter))]
        // [HasPermission(PermissionEnum.CanGetAllAttendance)]
        [HttpPost("GetAllAttendance")]
        public ApiBaseResponse<List<MemberAttendance>> GetAllAttendance(
            GetAllAttendanceRequest request)
        {
            return _backOfficeService.GetAllAttendance(request);
        }
        
        [HasPermission(PermissionEnum.CanGetAttendanceById)]
        [HttpPost("GetAttendanceById")]
        public ApiBaseResponse<List<MemberAttendance>> GetAttendanceById(
            GetAttendanceByIdRequest request)
        {
            return _backOfficeService.GetAttendanceById(request);
        }
        
        [HasPermission(PermissionEnum.CanGetAllRole)]
        [HttpGet("GetAllRole")]
        public ApiBaseResponse<List<RoleResponse>> GetAllRole()
        {
            return _backOfficeService.GetAllRole();
        }
        
        [HasPermission(PermissionEnum.CanAddMemberRole)]
        [HttpPost("AddMemberRole")]
        public ApiBaseResponse<string> AddMemberRole(AddMemberRoleRequest request)
        {
            return _backOfficeService.AddMemberRole(request);
        }
        
        [HasPermission(PermissionEnum.CanDeleteMemberRole)]
        [HttpPost("DeleteMemberRole")]
        public ApiBaseResponse<string> DeleteMemberRole(DeleteMemberRoleRequest request)
        {
            return _backOfficeService.DeleteMemberRole(request);
        }
        
        [HasPermission(PermissionEnum.CanGetAllPermission)]
        [HttpGet("GetAllPermission")]
        public ApiBaseResponse<Dictionary<string,List<GetAllPermissionResponse>>> GetAllMemberRole()
        {
            return _backOfficeService.GetAllPermission();
        }
        

        [HasPermission(PermissionEnum.CanAddRole)]
        [HttpPost("AddRole")]
        public ApiBaseResponse<string> AddRole(AddRoleRequest request)
        {
            return _backOfficeService.AddRole(request);
        }
        
        [HasPermission(PermissionEnum.CanGetPermissionByRoleId)]
        [HttpPost("GetPermissionByRoleId")]
        public ApiBaseResponse<List<int>> GetPermissionByRoleId(GetPermissionByIdRequest request)
        {
            return _backOfficeService.GetPermissionByRoleId(request);
        }
        
        [HasPermission(PermissionEnum.CanUpdateRolePermission)]
        [HttpPost("UpdateRolePermission")]
        public ApiBaseResponse<string> UpdateRolePermission(UpdateRolePermissionRequest request)
        {
            return _backOfficeService.UpdateRolePermission(request);
        }
        
        [HasPermission(PermissionEnum.CanUpdateMemberRole)]
        [HttpPost("UpdateMemberRole")]
        public ApiBaseResponse<string> UpdateMemberRole(UpdateMemberRoleRequest request)
        {
            return _backOfficeService.UpdateMemberRole(request);
        }
        
        [HasPermission(PermissionEnum.CanUpdateRoleByMemberId)]
        [HttpPost("UpdateRoleByMemberId")]
        public ApiBaseResponse<string> UpdateRoleByMemberId(UpdateRoleByMemberIdRequest request)
        {
            return _backOfficeService.UpdateRoleByMemberId(request);
        }
        
        [HasPermission(PermissionEnum.CanGetRoleByMemberId)]
        [HttpPost("GetRoleByMemberId")]
        public ApiBaseResponse<List<RoleResponse>> GetRoleByMemberId(GetRoleByMemberIdRequest request)
        {
            return _backOfficeService.GetRoleByMemberId(request);
        }
        
        [HasPermission(PermissionEnum.CanGetLeaveRequestsByMemberId)]
        [HttpPost("GetLeaveRequestsByMemberId")]
        public ApiBaseResponse<List<TakeLeave>> GetLeaveRequestsByMemberId(GetLeaveRequestsByMemberIdRequest request)
        {
            return _backOfficeService.GetLeaveRequestsByMemberId(request);
        }

        [HasPermission(PermissionEnum.CanRegisterLeaveAmount)]
        [HttpPost("RegisterLeaveAmount")]
        public ApiBaseResponse<RepositoryBaseResponse> RegisterLeaveAmount(ByMemberIdRequest request)
        {
            return _backOfficeService.RegisterLeaveAmount(request);
        }
        
        [HasPermission(PermissionEnum.CanUpdateMemberMonthlyTransferStatus)]
        [HttpPost("UpdateMemberMonthlyTransferStatus")]
        public ApiBaseResponse<RepositoryBaseResponse> UpdateMemberMonthlyTransferStatus(UpdateMemberMonthlyTransferStatusRequest request)
        {
            return _backOfficeService.UpdateMemberMonthlyTransferStatus(request);
        }

        [HasPermission(PermissionEnum.CanBatchUpdateMonthlyTransferStatus)]
        [HttpPost("BatchUpdateMonthlyTransferStatus")]
        public ApiBaseResponse<RepositoryBaseResponse> BatchMemberMonthlyTransferStatus(BatchUpdateMonthlyTransferStatusRequest request)
        {
            return _backOfficeService.BatchUpdateMonthlyTransferStatus(request);
        }
        
        [HasPermission(PermissionEnum.CanGetAllPositionByTeamId)]
        [HttpPost("GetAllPositionByTeamId")]
        public ApiBaseResponse<List<PositionResponse>> GetAllPositionByTeamId(GetPositionByTeamIdRequest request)
        {
            return _backOfficeService.GetAllPositionByTeamId(request);
        }
        
        [HasPermission(PermissionEnum.CanGetMembers)]
        [HttpPost("GetResignTransfer")]
        public ApiBaseResponse<ResignTransferResponse> GetResignTransfer(GetResignTransferRequest request)
        {
            return _backOfficeService.GetResignTransfer(request);
        }
        
        
        [HasPermission(PermissionEnum.CanAddMonthlyTransfer)]
        [HttpPost("AddResignTransfer")]
        public ApiBaseResponse<string> AddResignTransfer(AddResignTransfer request)
        {
            return _backOfficeService.AddResignTransfer(request);
        }
        
        [HasPermission(PermissionEnum.CanGetAllDepartments)]
        [HttpGet("GetAllDepartments")]
        public ApiBaseResponse<List<DepartmentResponse>> GetAllDepartments()
        {
            return _backOfficeService.GetAllDepartments();
        }
        
        [HasPermission(PermissionEnum.CanAddDepartment)]
        [HttpPost("AddDepartment")]
        public ApiBaseResponse<RepositoryBaseResponse> AddDepartment(AddDepartmentRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            request.CreatedBy = jwtData.Id;
            return _backOfficeService.AddDepartment(request);
        }
        
        [HasPermission(PermissionEnum.CanUpdateDepartment)]
        [HttpPost("UpdateDepartment")]
        public ApiBaseResponse<RepositoryBaseResponse> UpdateDepartment(UpdateDepartmentRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            request.ModifiedBy = jwtData.Id;
            return _backOfficeService.UpdateDepartment(request);
        }
        
        [HasPermission(PermissionEnum.CanDeleteDepartment)]
        [HttpDelete("DeleteDepartment")]
        public ApiBaseResponse<RepositoryBaseResponse> DeleteDepartment(int departmentId)
        {
            return _backOfficeService.DeleteDepartment(departmentId);
        }
        
        [HasPermission(PermissionEnum.CanAddTeam)]
        [HttpPost("AddTeam")]
        public ApiBaseResponse<RepositoryBaseResponse> AddTeam(AddTeamRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            request.CreatedBy = jwtData.Id;
            return _backOfficeService.AddTeam(request);
        }
        
        [HasPermission(PermissionEnum.CanUpdateTeam)]
        [HttpPost("UpdateTeam")]
        public ApiBaseResponse<RepositoryBaseResponse> UpdateTeam(UpdateTeamRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            request.ModifiedBy = jwtData.Id;
            return _backOfficeService.UpdateTeam(request);
        }
        
        [HasPermission(PermissionEnum.CanDeleteTeam)]
        [HttpDelete("DeleteTeam")]
        public ApiBaseResponse<RepositoryBaseResponse> DeleteTeam(int teamId)
        {
            return _backOfficeService.DeleteTeam(teamId);
        }
        
        [HasPermission(PermissionEnum.CanGetAllPosition)]
        [HttpGet("GetAllPosition")]
        public ApiBaseResponse<List<PositionResponse>> GetAllPosition()
        {
            return _backOfficeService.GetAllPosition();
        }
        
        [HasPermission(PermissionEnum.CanAddPosition)]
        [HttpPost("AddPosition")]
        public ApiBaseResponse<RepositoryBaseResponse> AddPosition(AddPositionRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            request.CreatedBy = jwtData.Id;
            return _backOfficeService.AddPosition(request);
        }
        
        [HasPermission(PermissionEnum.CanUpdatePosition)]
        [HttpPost("UpdatePosition")]
        public ApiBaseResponse<RepositoryBaseResponse> UpdatePosition(UpdatePositionRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            request.ModifiedBy = jwtData.Id;
            return _backOfficeService.UpdatePosition(request);
        }
        
        [HasPermission(PermissionEnum.CanDeletePosition)]
        [HttpDelete("DeletePosition")]
        public ApiBaseResponse<RepositoryBaseResponse> DeletePosition(int positionId)
        {
            return _backOfficeService.DeletePosition(positionId);
        }
        
        [HasPermission(PermissionEnum.CanAddJobGrade)]
        [HttpPost("AddJobGrade")]
        public ApiBaseResponse<RepositoryBaseResponse> AddJobGrade(AddJobGradeRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            request.CreatedBy = jwtData.Id;
            return _backOfficeService.AddJobGrade(request);
        }
        
        [HasPermission(PermissionEnum.CanUpdateJobGrade)]
        [HttpPost("UpdateJobGrade")]
        public ApiBaseResponse<RepositoryBaseResponse> UpdateJobGrade(UpdateJobGradeRequest request)
        {
            var jwtData = (JwtData)HttpContext.Items["JwtData"];
            request.ModifiedBy = jwtData.Id;
            return _backOfficeService.UpdateJobGrade(request);
        }
        
        [HasPermission(PermissionEnum.CanDeleteJobGrade)]
        [HttpDelete("DeleteJobGrade")]
        public ApiBaseResponse<RepositoryBaseResponse> DeleteJobGrade(int jobGradeId)
        {
            return _backOfficeService.DeleteJobGrade(jobGradeId);
        }
    }

}