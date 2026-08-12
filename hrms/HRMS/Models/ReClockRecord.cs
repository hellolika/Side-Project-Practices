using HRMS.Enum;

namespace HRMS.Models
{
    public class ReClockRecord
    {
		public int RequestId { get; set; }
		public int MemberId { get; set; }
		public string Username { get; set; }
		public string TeamName { get; set; }
		public DateTime Date { get; set; }
		public TimeSpan Time { get; set; }
		public bool IsClockIn { get; set; }
		public string Reason { get; set; }
		public string Location { get; set; }
		public StatusEnum Status { get; set; }
	}
}
