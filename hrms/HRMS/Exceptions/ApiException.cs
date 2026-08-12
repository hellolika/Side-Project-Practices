using HRMS.Enum;

namespace HRMS.Exceptions
{
    public class ApiException : Exception
    {
        public ApiErrorEnum Error { get; set; }

        public ApiException(ApiErrorEnum code, string message) : base(message)
        {
            Error = code;
        }
    }
}
