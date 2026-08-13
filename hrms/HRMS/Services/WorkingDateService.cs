using System.Diagnostics.CodeAnalysis;
using HRMS.Services.Interfaces;

namespace HRMS.Services
{
    public class WorkingDateService : IWorkingDateService
    {
        public List<DateTime> GetWorkingDays(DateTime startDate, DateTime endDate)
        {
            var workingDays = new List<DateTime>();

            for (DateTime i = startDate; i <= endDate; i = i.AddDays(1))
            {
                if (IsWorkingDate(i))
                {
                    workingDays.Add(i);
                }
            }

            return workingDays;
        }

        public decimal GetNumberOfLeavesDays(DateTimeOffset StartDate, DateTimeOffset EndDate)
        {
            decimal numberOfDays = GetWorkingDays(StartDate.Date, EndDate.Date).Count;
            var twelvePM = TimeSpan.FromHours(12);
            var onePM = TimeSpan.FromHours(13);
            if ((StartDate.TimeOfDay >= twelvePM && EndDate.TimeOfDay > twelvePM) ||
                (StartDate.TimeOfDay < twelvePM && EndDate.TimeOfDay <= onePM))
            {
                numberOfDays -= 0.5M;
            }
            else if (StartDate.TimeOfDay >= twelvePM && EndDate.TimeOfDay <= onePM)
            {
                numberOfDays -= 1M;
            }

            return numberOfDays;
        }

        public IEnumerable<LeavePeriod> GetLeavePeriods(DateTimeOffset startDate, DateTimeOffset endDate)
        {
            var result = new List<LeavePeriod>();

            var currentStart = startDate;
            while (currentStart.Date <= endDate.Date)
            {
                DateTimeOffset currentEnd;
                if (currentStart.Date == endDate.Date)
                {
                    currentEnd = endDate;
                }
                else
                {
                    currentEnd = currentStart.Date.AddHours(17).AddMinutes(30);
                }

                result.Add(new LeavePeriod
                {
                    StartDate = currentStart,
                    EndDate = currentEnd
                });

                currentStart = currentStart.Date.AddDays(1).AddHours(8).AddMinutes(30);
            }

            var groupedLeavePeriods = result
                .GroupBy(lp => $"{lp.StartDate.Year}-{lp.StartDate.Month:D2}")
                .ToDictionary(
                    g => g.Key,
                    g => new LeavePeriod()
                    {
                        StartDate = g.Min(lp => lp.StartDate),
                        EndDate = g.Max(lp => lp.EndDate)
                    });

            return groupedLeavePeriods.Values.ToList();
        }

        public bool IsWorkingDate(DateTime date)
        {
            return date.DayOfWeek != DayOfWeek.Saturday && date.DayOfWeek != DayOfWeek.Sunday;
        }
    }
}