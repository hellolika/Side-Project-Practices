using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class DoReClockRequest
    {
        [JsonProperty("Date")]
        public DateTimeOffset? Date { get; set; }

        [JsonProperty("Time")]
        public TimeSpan? Time { get; set; }

        [JsonProperty("Location")]
        public string Location { get; set; }

        [JsonProperty("Reason")]
        public string Reason { get; set; }

        [JsonProperty("IsClockIn")]
        public bool IsClockIn { get; set; }

        [JsonIgnore]
        public int MemberId { get; set; }

        public void CheckModels()
        {
            if(Date == null)
            {
                ThrowExpection("Date is required");
            }
            if (Time == null)
            {
                ThrowExpection("Time is required");
            }
            if (string.IsNullOrWhiteSpace(Location))
            {
                ThrowExpection("Location is required");
            }            
            if (Location.Length > 200)
            {
                ThrowExpection("Maximum characters allowed for location is 200");
            }
            if (string.IsNullOrWhiteSpace(Reason))
            {
                ThrowExpection("Reason is required");
            }
            if (Reason.Length > 200)
            {
                ThrowExpection("Maximum characters allowed for reason is 200");
            }
        }

        private static void ThrowExpection(string message)
        {
            throw new ApiException(ApiErrorEnum.InvalidModelState, message);
        }
    }
}