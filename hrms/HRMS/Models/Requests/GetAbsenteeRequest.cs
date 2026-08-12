using Newtonsoft.Json;
using System.Data;

namespace HRMS.Models.Requests
{
    public class GetAbsenteeRequest
    {
        [JsonProperty("StartDate")]
        public DateTimeOffset StartDate { get; set; }

        [JsonProperty("EndDate")]
        public DateTimeOffset EndDate { get; set; }
    }
    
}
