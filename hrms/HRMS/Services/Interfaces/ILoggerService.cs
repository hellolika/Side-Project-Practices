namespace HRMS.Services.Interfaces
{
    public interface ILoggerService
    {
        void Debug(string message);

        void Error(string message);

        void Info(string message);
    }
}