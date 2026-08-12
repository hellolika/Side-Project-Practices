namespace HRMS.Models
{
    public class Attendance
    {
        public int Id { get; set; }
        public int MemberId { get; set; }
        public DateTime WorkDate { get; set; }
        public TimeSpan? ClockIn { get; set; }
        public TimeSpan? ClockOut { get; set; }
        public string? ClockInLocation { get; set; }
        public string? ClockOutLocation { get; set; }
    }
}
