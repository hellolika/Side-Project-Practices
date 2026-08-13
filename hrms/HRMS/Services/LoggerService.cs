using HRMS.Services.Interfaces;

namespace HRMS.Services
{
    public class LoggerService : ILoggerService
    {
        private readonly ILogger<LoggerService> _loggerService;

        public LoggerService(ILogger<LoggerService> loggerService)
        {
            _loggerService = loggerService;
        }

        public void Debug(string message)
        {
            _loggerService.LogDebug($"{message}");
        }

        public void Info(string message)
        {
            _loggerService.LogInformation($"{message}");
        }

        public void Error(string message)
        {
            _loggerService.LogError($"{message}");
        }
    }
}