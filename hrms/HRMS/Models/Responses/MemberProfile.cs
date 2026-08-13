using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class MemberProfile: Member
    {
        [JsonIgnore]
        public override string Password { get; set; }

        [JsonProperty("TeamName")]
        public string TeamName { get; set; }

        [JsonProperty("WorkLocation")]
        public string WorkLocation { get; set; }

        [JsonProperty("Permissions")]
        public List<GetAllPermissionResponse> Permissions { get; set; }

    }
}