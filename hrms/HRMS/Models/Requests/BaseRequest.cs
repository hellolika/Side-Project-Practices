using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class BaseRequest
    {
        [JsonProperty("TimeZone")]
        [JsonConverter(typeof(CustomTimeSpanConverter))]
        public TimeSpan? TimeZone { get; set; }

        public virtual void CheckModels()
        {
            if (TimeZone is null)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "TimeZone is required");
            }
        }
    }
}
