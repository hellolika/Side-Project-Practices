using System;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
	public class UpdateRoleByMemberIdRequest
	{
        [JsonProperty("MemberId")]
        public int MemberId { get; set; }

        [JsonProperty("RoleList")]
        public List<int> RoleList { get; set; }

        [JsonIgnore]
        public int CreatedBy { get; set; }
    }
}

