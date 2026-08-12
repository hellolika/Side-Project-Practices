namespace HRMS.Services.Interfaces
{
    public interface IWorkingDateService
    {
        bool IsWorkingDate(DateTime date);

        List<DateTime> GetWorkingDays(DateTime startDate, DateTime endDate);

        decimal GetNumberOfLeavesDays(DateTimeOffset startDate, DateTimeOffset endDate);
        IEnumerable<LeavePeriod> GetLeavePeriods(DateTimeOffset startDate, DateTimeOffset endDate);
    }

    public class LeavePeriod
    {
        public DateTimeOffset StartDate { get; set; }
        public DateTimeOffset EndDate { get; set; }
    }
}