using HRMS.Enum;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class LoginResponse : RepositoryBaseResponse
    {
        [JsonIgnore]
        public override ApiErrorEnum ErrorCode { get; set; }

        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        [JsonProperty("Jwt")]
        public string Jwt { get; set; }

        [JsonProperty("Username")]
        public string Username { get; set; } = string.Empty;

        [JsonProperty("Email")]
        public string Email { get; set; } = string.Empty;

        [JsonProperty("PhoneNumber")]
        public string PhoneNumber { get; set; }

        [JsonProperty("Address")]
        public string Address { get; set; } = string.Empty;

        [JsonIgnore]
        public int Permission { get; set; }

        [JsonProperty("Role")]
        public List<int> Role { get; set; }

        [JsonProperty("BankAccount")]
        public string BankAccount { get; set; } = string.Empty;

        [JsonProperty("Salary")]
        public int Salary { get; set; }

        [JsonProperty("IsInProbation")]
        public bool IsInProbation { get; set; }
        
        [JsonProperty("Remark")]
        public string Remark { get; set; } = string.Empty;
        
        [JsonProperty("BankName")]
        public string BankName { get; set; } = string.Empty;
        
        [JsonProperty("JoinDate")] 
        public DateTimeOffset? JoinDate { get; set; }

        [JsonProperty("TeamId")]
        public int TeamId { get; set; }

        [JsonProperty("TeamName")]
        public string TeamName { get; set; }

        [JsonProperty("JobGrade")]
        public int JobGrade { get; set; }

        [JsonProperty("Position")]
        public string Position { get; set; }

        [JsonProperty("IsFirstTimeUser")]
        public bool IsFirstTimeUser { get; set; }

        [JsonProperty("Permissions")]
        public List<GetAllPermissionResponse> Permissions { get; set; }
        
        [JsonProperty("IsCanSeeMemberSalary")]
        public bool IsCanSeeMemberSalary { get; set; }

        public LoginResponse()
        {
        }
    }
}