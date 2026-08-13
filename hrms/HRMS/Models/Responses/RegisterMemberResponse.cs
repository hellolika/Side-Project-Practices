using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
    public class RegisterMemberResponse : RepositoryBaseResponse
    {
        [JsonProperty("MemberId")]
        public int MemberId { get; set; }
    }
}
