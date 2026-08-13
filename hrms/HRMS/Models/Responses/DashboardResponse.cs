using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class DashboardResponse : RepositoryBaseResponse
    {
        [JsonProperty("Employees")]
        public int Employees { get; set; }

        [JsonProperty("Departments")]
        public int Departments { get; set; }

        [JsonProperty("LeaveRequests")]
        public int LeaveRequests { get; set; }

        [JsonProperty("TotalLeaveToday")]
        public int TotalLeaveToday { get; set; }

        [JsonProperty("ProbationEmployees")]
        public int ProbationEmployees { get; set; }

        [JsonProperty("PermanentEmployees")]
        public int PermanentEmployees { get; set; }
        
    }
}
