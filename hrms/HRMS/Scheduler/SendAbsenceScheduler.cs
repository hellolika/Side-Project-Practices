using HRMS.Models.Requests;
using HRMS.Models.Responses;
using HRMS.Repositories.Interfaces;
using HRMS.Scheduler.Interfaces;
using HRMS.Services.Interfaces;

namespace HRMS.Scheduler;

public class SendAbsenceScheduler : IScheduler
{
    private readonly ILoggerService _loggerService;
    private readonly ISlackService _slackService;

    public SendAbsenceScheduler(ILoggerService loggerService, ISlackService slackService)
    {
        _loggerService = loggerService;
        _slackService = slackService;
    }

    public void RunJob()
    {
        _loggerService.Info($"Scheduled to send absence alert to all member on {DateTime.Now}");
        _slackService.SendAbsenceAlert();
    }
}