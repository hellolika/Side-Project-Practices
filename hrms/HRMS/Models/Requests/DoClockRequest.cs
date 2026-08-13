using HRMS.Enum;
using HRMS.Exceptions;
using HRMS.Extensions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class DoClockRequest : BaseRequest
    {
        [JsonIgnore]
        public DateTimeOffset Date { get; set; }

        [JsonProperty("IsClockIn")]
        public bool IsClockIn { get; set; }

        [JsonProperty("IsInCompany")]
        public bool IsInCompany { get; set; }

        [JsonProperty("Location")]
        public string Location { get; set; }

        [JsonProperty("ClockInRemark")] public string ClockInRemark { get; set; } = string.Empty;

        [JsonProperty("ClockOutRemark")] public string ClockOutRemark { get; set; } = string.Empty;
        
        [JsonIgnore]
        public int MemberId { get; set; }

        public override void CheckModels()
        {
            base.CheckModels();   
            if(!IsInCompany && string.IsNullOrWhiteSpace(Location))
            {
                throw new ApiException(ApiErrorEnum.InvalidModelState, "Location is required when not within company");
            }
        }
    }
}