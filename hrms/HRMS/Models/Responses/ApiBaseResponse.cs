using HRMS.Enum;
using HRMS.Extensions;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class ApiBaseResponse<TResult>
    {
        [JsonProperty("ErrorCode")]
        public ApiErrorEnum ErrorCode { get; set; }

        [JsonProperty("ErrorMessage")]
        public string ErrorMessage { get; set; }

        [JsonProperty("Result")]
        public TResult Result { get; set; }

        public ApiBaseResponse(ApiErrorEnum errorCode)
        {
            ErrorCode = errorCode;
            ErrorMessage = errorCode.GetDescription();
            Result = default;
        }

        public ApiBaseResponse(ApiErrorEnum errorCode, string message)
        {
            ErrorCode = errorCode;
            ErrorMessage = message;
            Result = default;
        }

        public ApiBaseResponse()
        {
            ErrorMessage = ErrorCode.GetDescription();
        }

        public bool IsSuccess() => ErrorCode == ApiErrorEnum.NoError;

        public ApiBaseResponse(TResult result)
        {
            Result = result;
            ErrorCode = ApiErrorEnum.NoError;
            ErrorMessage = ApiErrorEnum.NoError.ToString();
        }
    }
}