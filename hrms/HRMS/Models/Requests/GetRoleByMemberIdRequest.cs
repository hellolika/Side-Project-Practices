using System;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
	public class GetRoleByMemberIdRequest
	{
        [JsonProperty("MemberId")] public int MemberId { get; set; }
    }
}

