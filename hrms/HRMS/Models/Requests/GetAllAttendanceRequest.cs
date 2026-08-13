using HRMS.Enum;
using HRMS.Exceptions;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class GetAllAttendanceRequest : BaseRequest
    {
        [JsonProperty("Date")]
        public DateTimeOffset Date { get; set; }

    }
}
