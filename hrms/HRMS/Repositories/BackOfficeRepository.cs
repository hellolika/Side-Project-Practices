using HRMS.Models;
using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Models.Settings;
using HRMS.Repositories.Interfaces;
using HRMS.Services.Interfaces;
using Microsoft.Extensions.Options;
using System.Data;
using DBCiper.Repositories;

namespace HRMS.Repositories
{
    public class BackOfficeRepository : BaseRepository, IBackOfficeRepository
    {
        private BackOfficeStoreProcedureSettings _storeProcedureSettings;

        public BackOfficeRepository(ILoggerService loggerService, IConfiguration configuration,
            IOptionsMonitor<BackOfficeStoreProcedureSettings> storeProcedureSettings)
            : base(configuration)
        {
            _storeProcedureSettings = storeProcedureSettings.CurrentValue;
            storeProcedureSettings.OnChange(newValue => { _storeProcedureSettings = newValue; });
        }

        public IEnumerable<GetAbsenteeResponse> GetAbsentee(DataTable tvpWorkingDay)
        {
            return Query<GetAbsenteeResponse>(_storeProcedureSettings.GetAllAbsentee, new
            {
                tvpWorkingDay
            });
        }

        public RepositoryBaseResponse LeaveRequestApproval(RequestApprovalRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.LeaveRequestApproval, new
            {
                approverId = request.ApproverId,
                requestId = request.RequestId,
                isApproved = request.IsApproved,
                responseReason = request.ResponseReason
            }).First();
        }

        public RepositoryBaseResponse ReClockRequestApproval(RequestApprovalRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.ReClockRequestApproval, new
            {
                approverId = request.ApproverId,
                requestId = request.RequestId,
                isApproved = request.IsApproved,
                responseReason = request.ResponseReason
            }).First();
        }

        public List<GetMembersResponse> GetMembers()
        {
            return Query<GetMembersResponse>(_storeProcedureSettings.GetAllMembers).ToList();
        }

        public RepositoryBaseResponse AddNewLocation(AddLocationRequest location)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddLocation,
                new
                {
                    location.LocationName,
                    location.Latitude,
                    location.Longitude,
                    location.Range
                }).First();
        }

        public List<LeavesResponse> GetAllLeaves(TimeRangeRequest request)
        {
            return Query<LeavesResponse>(_storeProcedureSettings.GetAllLeaves,
                new
                {
                    startDate = request.StartDate.Date,
                    endDate = request.EndDate.Date
                }).ToList();
        }

        public List<ReClocksResponse> GetAllReClock(TimeRangeRequest request)
        {
            return Query<ReClocksResponse>(_storeProcedureSettings.GetAllReClock,
                new
                {
                    startDate = request.StartDate.Date,
                    endDate = request.EndDate.Date
                }).ToList();
        }

        public RegisterMemberResponse RegisterMember(RegisterMemberRequest request)
        {
            return Query<RegisterMemberResponse>(_storeProcedureSettings.RegisterMember,
                new
                {
                    username = request.Username,
                    email = request.Email,
                    password = request.Password,
                    phoneNumber = request.PhoneNumber,
                    gender = request.Gender,
                    address = request.Address,
                    salary = request.Salary,
                    permission = request.Permission,
                    bankAccount = request.BankAccount,
                    isInProbation = request.IsInProbation,
                    remark = request.Remark,
                    teamId = request.TeamId,
                    jobGrade = request.JobGrade,
                    workLocationId = request.WorkLocationId,
                    joinDate = request.JoinDate,
                    isAlertProbation = request.IsAlertProbation,
                    bankName = request.BankName,
                    positionId = request.PositionId,
                    position = request.Position,
                    employeeId = request.EmployeeId,
                    birthday = request.Birthday,
                    nationalId = request.NationalId,
                    vehicleType = request.VehicleType,
                    vehicleNumber = request.VehicleNumber,
                    departmentId = request.DepartmentId,
                    isManager = request.IsManager,
                    isCanSeeMemberSalary = request.IsCanSeeMemberSalary
                }).First();
        }

        public RepositoryBaseResponse EditMember(EditMemberInfoRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.EditMember,
                new
                {
                    memberId = request.MemberId,
                    username = request.Username,
                    email = request.Email,
                    phoneNumber = request.PhoneNumber,
                    gender = request.Gender,
                    address = request.Address,
                    salary = request.Salary,
                    permission = request.Permission,
                    bankAccount = request.BankAccount,
                    isInProbation = request.IsInProbation,
                    isResigned = request.IsResigned,
                    remark = request.Remark,
                    teamId = request.TeamId,
                    jobGrade = request.JobGrade,
                    workLocationId = request.WorkLocationId,
                    joinDate = request.JoinDate,
                    isAlertProbation = request.IsAlertProbation,
                    bankName = request.BankName,
                    positionId = request.PositionId,
                    position = request.Position,
                    employeeId = request.EmployeeId,
                    birthday = request.Birthday,
                    nationalId = request.NationalId,
                    vehicleType = request.VehicleType,
                    vehicleNumber = request.VehicleNumber,
                    departmentId = request.DepartmentId,
                    isManager = request.IsManager,
                    isCanSeeMemberSalary = request.IsCanSeeMemberSalary
                }).First();
        }

        public RepositoryBaseResponse DeleteMember(DeleteMemberRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeleteMember,
                new
                {
                    memberId = request.MemberId
                }).First();
        }

        public RepositoryBaseResponse ResetPassword(PasswordBaseRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.ResetMemberPassword,
                new
                {
                    memberId = request.MemberId,
                    Password = request.NewPassword
                }).First();
        }

        public List<MemberLeaveAmount> GetAllLeaveAmount()
        {
            return Query<MemberLeaveAmount>(_storeProcedureSettings.GetAllLeaveAmount).ToList();
        }

        public RepositoryBaseResponse UpdateLeaveAmount(UpdateLeaveAmountRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateLeaveAmount,
                new
                {
                    modifierId = request.ModifierId,
                    memberId = request.MemberId,
                    leaveAmount = request.LeaveAmount,
                    leavetype = request.LeaveType,
                    year = request.Year,
                }).First();
        }

        public RepositoryBaseResponse PopulateDefaultLeaves()
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.PopulateDefaultLeaves).First();
        }

        public RepositoryBaseResponse PopulateMemberRoleRecord(PopulateMemberRoleRecordRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.PopulateMemberRoleRecord, new
            {
                roleId = request.RoleId
            }).First();
        }

        public RepositoryBaseResponse AddAnnouncement(AddAnnouncementRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddAnnouncement,
                new
                {
                    request.Title,
                    request.Message,
                    request.CreatedBy
                }).First();
        }

        public RepositoryBaseResponse EditAnnouncement(EditAnnouncementRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.EditAnnouncement,
                new
                {
                    request.Id,
                    request.Title,
                    request.Message,
                    request.ModifiedBy
                }).First();
        }

        public RepositoryBaseResponse DeleteAnnouncement(int id)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeleteAnnouncement, new {id}).First();
        }

        public RepositoryBaseResponse AddTransferType(AddTransferTypeRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddTransferType, new
            {
                transferName = request.TransferName,
                isEnable = request.IsEnable,
                createdBy = request.CreateBy,
                modifiedBy = request.ModifiedBy
            }).First();
        }

        public List<TransferTypeResponse> GetAllTransferTypes()
        {
            return Query<TransferTypeResponse>(_storeProcedureSettings.GetAllTransferTypes).ToList();
        }

        public List<TransferResponse> GenerateMemberMonthlyTransfer(GenerateMemberMonthlyTransferRequest request)
        {
            return Query<TransferResponse>(_storeProcedureSettings.GenerateMemberMonthlyTransfer, new
            {
                startDate = request.StartDate.Date,
                endDate = request.EndDate.Date,
            }).ToList();
        }

        public RepositoryBaseResponse AddMonthlyTransfer(AddMonthlyTransferRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddMonthlyTransfer, new
            {
                transferId = request.TransferId,
                memberId = request.MemberId,
                propertyTypeId = request.TransferTypeId,
                amount = request.Amount,
                dayCount = request.DayCount,
                status = request.Status,
                remark = request.Remark,
                payRollDate = request.PayStartDate.DateTime,
                startDate = request.PayStartDate.DateTime,
                endDate = request.PayEndDate.DateTime,
                createdBy = request.CreateBy,
                modifiedBy = request.ModifiedBy
            }).First();
        }

        public RepositoryBaseResponse DeleteMonthlyTransfer(DeleteMonthlyTransferRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeleteMonthlyTransfer,
                new {transferId = request.TransferId}).First();
        }

        public List<TransferResponse> GetMonthlyTransfers(GenerateMemberMonthlyTransferRequest request)
        {
            return Query<TransferResponse>(_storeProcedureSettings.GetMonthlyTransfers, new
            {
                startDate = request.StartDate.Date,
                endDate = request.EndDate.Date,
            }).ToList();
        }

        public RepositoryBaseResponse AddOrEditLeaveRecord(AddOrEditLeaveRecordRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddOrEditLeaveRecord, new
            {
                leaveRecordId = request.LeaveRecordId,
                memberId = request.MemberId,
                numberOfDay = request.NumberOfDay,
                startDate = request.StartDate,
                endDate = request.EndDate,
                leaveType = request.LeaveType,
            }).First();
        }

        public RepositoryBaseResponse AddUnpaidLeave(int memberId)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddUnpaidLeave, new
            {
                memberId = memberId,
            }).First();
        }

        public RepositoryBaseResponse EditMemberLeaves(EditMemberLeaveRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.EditMemberLeaves, new
            {
                memberId = request.MemberId,
                leaveTypeId = request.LeaveTypeId,
                readjustAmount = request.ReadjustAmount
            }).First();
        }

        public List<LeaveAmountResponse> GetLeaveAmountByMemberId(GetLeaveAmountByMemberIdRequest request)
        {
            return Query<LeaveAmountResponse>(_storeProcedureSettings.GetMemberLeaveAmount,
                new
                {
                    memberId = request.MemberId
                }).ToList();
        }

        public List<MemberAttendance> GetAllAttendance(GetAllAttendanceRequest request)
        {
            return Query<MemberAttendance>(_storeProcedureSettings.GetAllAttendance, new
            {
                date = request.Date
            }).ToList();
        }

        public List<RoleResponse> GetAllRole()
        {
            return Query<RoleResponse>(_storeProcedureSettings.GetAllRole).ToList();
        }

        public RepositoryBaseResponse AddMemberRole(AddMemberRoleRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddMemberRole, new
            {
                memberId = request.MemberId,
                roleId = request.RoleId,
                createdBy = request.CreatedBy
            }).First();
        }

        public RepositoryBaseResponse DeleteMemberRole(DeleteMemberRoleRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeleteMemberRole, new
            {
                memberId = request.MemberId,
                roleId = request.RoleId,
            }).First();
        }

        public List<GetAllPermissionResponse> GetAllPermission()
        {
            return Query<GetAllPermissionResponse>(_storeProcedureSettings.GetAllPermission).ToList();
        }

        public DashboardResponse GetDashboard()
        {
            return Query<DashboardResponse>(_storeProcedureSettings.GetDashboard, new
            {
            }).First();
        }

        public RepositoryBaseResponse AddRole(AddRoleRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddRole, new
            {
                roleName = request.RoleName,
                roleDescription = request.RoleDescription,
                createdBy = request.CreatedBy
            }).First();
        }

        public List<GetPermissionByIdRepsonse> GetPermissionByRoleId(GetPermissionByIdRequest request)
        {
            return Query<GetPermissionByIdRepsonse>(_storeProcedureSettings.GetPermissionByRoleId, new
            {
                roleId = request.RoleId
            }).ToList();
        }

        public RepositoryBaseResponse UpdateRolePermission(UpdateRolePermissionRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateRolePermission, new
            {
                roleId = request.RoleId,
                permissionList = "[" + string.Join(",", request.PermissionList) + "]"
            }).First();
        }

        public RepositoryBaseResponse UpdateMemberRole(UpdateMemberRoleRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateMemberRole, new
            {
                roleId = request.RoleId,
                memberList = "[" + string.Join(",", request.MemberList) + "]",
                createdBy = request.CreatedBy
            }).First();
        }

        public List<MemberRoleReponse> GetMemberByRoleId(int roleId)
        {
            return Query<MemberRoleReponse>(_storeProcedureSettings.GetMemberByRoleId, new {roleId}).ToList();
        }

        public RepositoryBaseResponse UpdateRoleByMemberId(UpdateRoleByMemberIdRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateRoleByMemberId, new
            {
                memberId = request.MemberId,
                roleList = "[" + string.Join(",", request.RoleList) + "]",
                createdBy = request.CreatedBy
            }).First();
        }


        public List<RoleResponse> GetRolesBy(int memberId)
        {
            return Query<RoleResponse>(_storeProcedureSettings.GetRoleByMemberId, new {memberId}).ToList();
        }

        public List<BeneficiaryList> GetBeneficiaryType()
        {
            return Query<BeneficiaryList>(_storeProcedureSettings.GetBeneficiaryType).ToList();
        }

        public List<MemberAttendance> GetAttendanceById(GetAttendanceByIdRequest request)
        {
            return Query<MemberAttendance>(_storeProcedureSettings.GetAttendanceById, new
            {
                memberId = request.MemberId,
                startDate = request.StartDate,
                endDate = request.EndDate
            }).ToList();
        }

        public List<TakeLeave> GetLeaveRequestsByMemberId(GetLeaveRequestsByMemberIdRequest request)
        {
            return Query<TakeLeave>(_storeProcedureSettings.GetLeaveRequestsByMemberId, new
            {
                memberId = request.MemberId,
                startDate = request.StartDate,
                endDate = request.EndDate,
            }).ToList();
        }

        public List<ReClockRecord> GetAllReClockRecordsById(GetAttendanceByIdRequest request)
        {
            return Query<ReClockRecord>(_storeProcedureSettings.GetAllReClockRecordsById, new
            {
                memberId = request.MemberId,
                startDate = request.StartDate,
                endDate = request.EndDate
            }).ToList();
        }

        public RepositoryBaseResponse RegisterLeaveAmount(ByMemberIdRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.RegisterLeaveAmount, new
            {
                memberId = request.MemberId
            }).First();
        }

        public RepositoryBaseResponse AddLeaveAmountToAllMember(int amount = 1)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddLeaveAmountToAllMember, new
            {
                amount = amount
            }).First();
        }

        public RepositoryBaseResponse UpdateMemberMonthlyTransferStatus(
            UpdateMemberMonthlyTransferStatusRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateMemberMonthlyTransferStatus, new
            {
                transferId = request.TransferId,
                status = request.Status,
            }).First();
        }

        public RepositoryBaseResponse BatchUpdateMonthlyTransferStatus(BatchUpdateMonthlyTransferStatusRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.BatchUpdateMonthlyTransferStatus, new
            {
                statusId = request.StatusId,
                transferIdList = "[" + string.Join(",", request.TransferIdList) + "]",
            }).First();
        }

        public List<PositionResponse> GetAllPositionByTeamId(GetPositionByTeamIdRequest request)
        {
            return Query<PositionResponse>(_storeProcedureSettings.GetAllPositionByTeamId, new
            {
                teamId = request.TeamId,
            }).ToList();
        }

        public ResignTransferResponse GetResignTransfer(GetResignTransferRequest request)
        {
            return Query<ResignTransferResponse>(_storeProcedureSettings.GetResignTransfer, new
            {
                memberId = request.MemberId,
                startDate = request.StartDate,
                resignDate = request.ResignDate,
            }).FirstOrDefault();
        }
        
        public RepositoryBaseResponse AddDepartment(AddDepartmentRequest request)
        {
            var response = Query<RepositoryBaseResponse>(_storeProcedureSettings.AddDepartment, new
            {
                departmentName = request.DepartmentName,
                createdBy = request.CreatedBy
            }).First();
            return response;
        }

        public RepositoryBaseResponse UpdateDepartment(UpdateDepartmentRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateDepartment, new
            {
                departmentId = request.DepartmentId,
                departmentName = request.DepartmentName,
                modifiedBy = request.ModifiedBy
            }).First();
        }

        public List<DepartmentResponse> GetAllDepartments()
        {
            return Query<DepartmentResponse>(_storeProcedureSettings.GetAllDepartment).ToList();
        }

        public RepositoryBaseResponse DeleteDepartment(int departmentId)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeleteDepartment, new
            {
                departmentId
            }).First();
        }
        
        public RepositoryBaseResponse AddTeam(AddTeamRequest request)
        {
            var response = Query<RepositoryBaseResponse>(_storeProcedureSettings.AddTeam, new
            {
                teamName = request.TeamName,
                departmentId = request.DepartmentId,
                startTime = request.StartTime,
                endTime = request.EndTime,
                totalHour = request.TotalHour,
                createdBy = request.CreatedBy
            }).First();
            return response;
        }

        public RepositoryBaseResponse UpdateTeam(UpdateTeamRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateTeam, new
            {
                teamId = request.TeamId,
                teamName = request.TeamName,
                departmentId = request.DepartmentId,
                startTime = request.StartTime,
                endTime = request.EndTime,
                totalHour = request.TotalHour,
                modifiedBy = request.ModifiedBy
            }).First();
        }

        public RepositoryBaseResponse DeleteTeam(int teamId)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeleteTeam, new
            {
                teamId
            }).First();
        }
        
        public RepositoryBaseResponse AddPosition(AddPositionRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddPosition, new
            {
                positionName = request.PositionName,
                teamId = request.TeamId,
                createdBy = request.CreatedBy
            }).First();
        }
        
        public RepositoryBaseResponse UpdatePosition(UpdatePositionRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdatePosition, new
            {
                positionId = request.PositionId,
                positionName = request.PositionName,
                teamId = request.TeamId,
                modifiedBy = request.ModifiedBy
            }).First();
        }
        
        public RepositoryBaseResponse DeletePosition(int positionId)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeletePosition, new
            {
                positionId
            }).First();
        }
        
        public RepositoryBaseResponse AddJobGrade(AddJobGradeRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.AddJobGrade, new
            {
                jobGradeName = request.JobGradeName,
                positionId = request.PositionId,
                createdBy = request.CreatedBy
            }).First();
        }
        
        public RepositoryBaseResponse UpdateJobGrade(UpdateJobGradeRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateJobGrade, new
            {
                jobGradeId = request.JobGradeId,
                positionId = request.PositionId,
                jobGradeName = request.JobGradeName,
                modifiedBy = request.ModifiedBy
            }).First();
        }
        
        public RepositoryBaseResponse DeleteJobGrade(int jobGradeId)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeleteJobGrade, new
            {
                jobGradeId
            }).First();
        }
        
        public RepositoryBaseResponse UpdateLocation(UpdateLocationRequest request)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.UpdateLocation, new
            {
                locationId = request.Id,
                locationName = request.LocationName,
                latitude = request.Latitude,
                longitude = request.Longitude,
                range = request.Range,
                isEnabled = request.IsEnabled,
                modifiedBy = request.ModifiedBy
            }).First();
        }
        
        public RepositoryBaseResponse DeleteLocation(int locationId)
        {
            return Query<RepositoryBaseResponse>(_storeProcedureSettings.DeleteLocation, new
            {
                locationId
            }).First();
        }
        
        public List<PositionResponse> GetAllPosition()
        {
            return Query<PositionResponse>(_storeProcedureSettings.GetAllPosition, new {}).ToList();
        }
        
        public List<JobGradeResponse> GetAllJobGrade()
        {
            return Query<JobGradeResponse>(_storeProcedureSettings.GetAllJobGrade, new {}).ToList();
        }
    }
}