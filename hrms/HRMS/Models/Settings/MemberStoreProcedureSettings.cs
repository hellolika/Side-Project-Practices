namespace HRMS.Models.Settings
{
    public class MemberStoreProcedureSettings
    {
        public string Login { get; set; }
        public string DoClock { get; set; }
        public string CheckClockStatus { get; set; }
        public string DoReclock { get; set; }
        public string TakeLeave { get; set; }
        public string UpdateTakeLeave { get; set; }
        public string GetMemberLeaveAmount { get; set; }
        public string GetMemberLeaveAmountV2 { get; set; }
        public string GetLeaveRecordsByMemberId { get; set; }
        public string GetAllReClockRecords { get; set; }
        public string GetProfile { get; set; }
        public string UpdateProfile { get; set; }
        public string CancelLeave { get; set; }
        public string GetAllLeaveType { get; set; }
        public string GetAllTeamInfo { get; set; }
        public string GetLocation { get; set; }
        public string ChangePassword { get; set; }
        public string IsMemberExist { get; set; }
        public string GetSingleMemberAttendancesByDateRange { get; set; }
        public string GetSingleMemberReclockRecordsByDateRange { get; set; }
        public string GetMemberLeaveRequestByDates { get; set; }
        public string GetAnnouncements { get; set; }
        public string GetTotalAnnouncements { get; set; }
        public string GetMemberTransfers { get; set; }
        public string GetMemberPermission { get; set; }
        
        public string GetDepartmentManager { get; set; }
        
        public string UpsertSlackUsers { get; set; }
    }
}