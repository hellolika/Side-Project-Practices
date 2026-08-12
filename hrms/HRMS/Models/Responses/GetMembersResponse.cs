using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class GetMembersResponse : Member
    {
        [JsonIgnore]
        public override string Password { get; set; }

        [JsonProperty("TeamName")]
        public string TeamName { get; set; }

        [JsonProperty("WorkLocation")]
        public string WorkLocation { get; set; }
    }
}