using System;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
	public class UpdateMemberRoleRequest
	{
        [JsonProperty("RoleId")]
        public int RoleId { get; set; }

        [JsonProperty("MemberList")]
        public List<int> MemberList { get; set; }

        [JsonIgnore]
        public int CreatedBy { get; set; }
    }
}

