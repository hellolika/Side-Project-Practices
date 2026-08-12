using System;
using Newtonsoft.Json;

namespace HRMS.Models.Responses
{
	public class MemberRoleReponse
	{
        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        [JsonProperty("Username")]
        public string Username { get; set; }

        [JsonProperty("Email")]
        public string Email { get; set; }
    }
}

