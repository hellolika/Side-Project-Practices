using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class DeleteMemberRoleRequest
{
    [JsonProperty("MemberId")] public int MemberId { get; set; }

    [JsonProperty("RoleId")] public int RoleId { get; set; }
}