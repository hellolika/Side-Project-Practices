namespace HRMS.Models.Settings
{
    public class BackOfficeStoreProcedureSettings
    {
        public string GetAllAbsentee { get; set; }
        public string LeaveRequestApproval { get; set; }
        public string ReClockRequestApproval { get; set; }
        public string GetAllMembers { get; set; }
        public string AddLocation { get; set; }
        public string GetAllLeaves { get; set; }
        public string GetAllReClock { get; set; }
        public string RegisterMember { get; set; }
        public string EditMember { get; set; }
        public string DeleteMember { get; set; }
        public string ResetMemberPassword { get; set; }
        public string GetAllLeaveAmount { get; set; }
        public string UpdateLeaveAmount { get; set; }
        public string PopulateDefaultLeaves { get; set; }    
        public string PopulateMemberRoleRecord { get; set; }
        public string AddAnnouncement { get; set; }
        public string EditAnnouncement { get; set; }
        public string DeleteAnnouncement { get; set; }
        public string GetAllTransferTypes { get; set; }
        public string AddTransferType { get; set; }
        public string GenerateMemberMonthlyTransfer { get; set; }
        public string AddMonthlyTransfer { get; set; }
        public string DeleteMonthlyTransfer { get; set; }
        public string GetMonthlyTransfers { get; set; }
        public string AddOrEditLeaveRecord { get; set; }
        public string AddUnpaidLeave { get; set; }
        public string EditMemberLeaves { get; set; }
        public string GetMemberLeaveAmount { get; set; }
        public string GetAllAttendance { get; set; }
        public string GetAllRole { get; set; }
        public string AddMemberRole { get; set; }
        public string DeleteMemberRole { get; set; }
        public string GetAllPermission { get; set; }
        public string GetDashboard { get; set; }
        public string AddRole { get; set; }
        public string GetPermissionByRoleId { get; set; }
        public string UpdateRolePermission { get; set; }
        public string UpdateMemberRole { get; set; }
        public string GetMemberByRoleId { get; set; }
        public string UpdateRoleByMemberId { get; set; }
        public string GetRoleByMemberId { get; set; }
        public string GetBeneficiaryType { get; set; }
        public string GetAttendanceById { get; set; }
        public string GetLeaveRequestsByMemberId { get; set; }
        public  string GetAllReClockRecordsById { get; set; }
        public string RegisterLeaveAmount { get; set; }
        public string AddLeaveAmountToAllMember { get; set; }
        public string UpdateMemberMonthlyTransferStatus { get; set; }
        public string BatchUpdateMonthlyTransferStatus { get; set; }
        public string GetAllPositionByTeamId { get; set; }
        public string GetResignTransfer { get; set; }
        
        public string GetAllDepartment { get; set; }
        public string AddDepartment { get; set; }
        public string UpdateDepartment { get; set; }
        public string DeleteDepartment { get; set; }
        public string AddTeam { get; set; }
        public string UpdateTeam { get; set; }
        public string DeleteTeam { get; set; }
        public string AddPosition { get; set; }
        public string UpdatePosition { get; set; }
        public string DeletePosition { get; set; }
        
        public string AddJobGrade { get; set; }
        public string UpdateJobGrade { get; set; }
        public string DeleteJobGrade { get; set; }
        public string UpdateLocation { get; set; }
        public string DeleteLocation { get; set; }
        
        public string GetAllPosition { get; set; }
        
        public string GetAllJobGrade { get; set; }
    }
}
