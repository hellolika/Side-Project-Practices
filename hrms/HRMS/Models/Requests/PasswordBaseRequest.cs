using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
    public class PasswordBaseRequest
    {
        [JsonProperty("MemberId")]
        public virtual int MemberId { get; set; }

        [JsonProperty("NewPassword")]
        public string NewPassword { get; set; }
    }
}