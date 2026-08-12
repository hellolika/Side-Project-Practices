using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class GetAbsenteeResponse
    {
        [JsonProperty("MemberId")]
        public int Id { get; set; }

        [JsonProperty("Name")]
        public string Username { get; set; }

        [JsonProperty("TeamId")]
        public int TeamId { get; set; }

        [JsonProperty("TeamName")]
        public string TeamName { get; set; }

        [JsonProperty("WorkDate")]
        [JsonConverter(typeof(CustomDateTimeConverter))]
        public DateTime WorkDate { get; set; }

        [JsonProperty("LeaveId")]
        public int LeaveId { get; set; }         
        
        [JsonProperty("LeaveType")]
        public string LeaveType { get; set; } 
    }
}