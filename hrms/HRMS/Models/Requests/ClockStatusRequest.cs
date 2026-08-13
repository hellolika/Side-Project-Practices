using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class ClockStatusRequest
    {
        [JsonProperty("Date")]
        public DateTimeOffset? Date { get; set; }

        [JsonIgnore]
        public int MemberId { get; set; }

        public void CheckModels()
        {
            if(Date == null)
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Date is required");
            }
        }
    }
}