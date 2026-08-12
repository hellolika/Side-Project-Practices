using System;
using Newtonsoft.Json;

namespace HRMS.Models.Requests
{
	public class DeleteMemberRequest
	{
        [JsonProperty("MemberId")]
        public int MemberId { get; set; }
    }
}

