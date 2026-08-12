using HRMS.Repositories.Interfaces;
using HRMS.Scheduler.Interfaces;
using HRMS.Services.Interfaces;

namespace HRMS.Scheduler
{
    public class LeaveAmountScheduler : IScheduler
    {
        private readonly ILoggerService _loggerService;
        private readonly IBackOfficeRepository _backOfficeRepository;

        public LeaveAmountScheduler(IBackOfficeRepository backOfficeRepository, ILoggerService loggerService)
        {
            _backOfficeRepository = backOfficeRepository;
            _loggerService = loggerService;
        }

        public void RunJob()
        {
            _loggerService.Info("Scheduled to all leave amount to all member");
            // _backOfficeRepository.AddLeaveAmountToAllMember();
        }
    }
}
