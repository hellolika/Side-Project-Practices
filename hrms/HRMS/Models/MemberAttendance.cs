using HRMS.Models.Responses;
using HRMS.Repositories;
using Newtonsoft.Json;

namespace HRMS.Models
{
    public class MemberAttendance
    {
        [JsonProperty("MemberId")] 
        public int MemberId { get; set; }

        [JsonProperty("Username")]
        public string Username { get; set; }

        [JsonProperty("TeamName")]
        public string TeamName { get; set; }

        [JsonProperty("WorkDate")]
        public DateTime? WorkDate { get; set; }
        
        [JsonProperty("ClockInLocation")]
        public string ClockInLocation { get; set; }
        
        [JsonProperty("ClockOutLocation")]
        public string ClockOutLocation { get; set; }
        
        [JsonIgnore]
        public string Location { get; set; }
        
        [JsonIgnore]
        public string Reason { get; set; }
        
        [JsonProperty("ReClockInLocation")]
        public string ReClockInLocation { get; set; }
        
        [JsonProperty("ReClockOutLocation")]
        public string ReClockOutLocation { get; set; }
        
        [JsonProperty("ClockInRemark")]
        public string ClockInRemark { get; set; }
        
        [JsonProperty("ClockOutRemark")]
        public string ClockOutRemark { get; set; }
        
        [JsonProperty("ClockIn")] 
        public TimeSpan? ClockIn { get; set; }

        [JsonProperty("ClockOut")] 
        public TimeSpan? ClockOut { get; set; }

        [JsonProperty("ReClockIn")]
        public TimeSpan? ReClockIn { get; set; }

        [JsonProperty("ReClockOut")]
        public TimeSpan? ReClockOut { get; set; }
        
        [JsonProperty("Email")]
        public string Email { get; set; }
        
        public string SlackId { get; set; }
    
        public string SlackUsername { get; set; }    

        public string SlackRealName { get; set; }
        
    }
}
