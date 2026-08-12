using HRMS.Enum;

namespace HRMS.Models
{
    public class TakeLeaveRecord
    {
		public int RequestId { get; set; }
		public int MemberId { get; set; }
		public decimal NumberOfDay { get; set; }
		public DateTime StartDate { get; set; }
		public DateTime EndDate { get; set; }
		public int LeaveId { get; set; }
		public string LeaveType { get; set; }
		public string Reason { get; set; }
		public string ResponseReason { get; set; }
		public bool IsCancel { get; set; }
		public StatusEnum Status { get; set; }
		public DateTime SubmittedOn { get; set; }
	}
}
