using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class RoleResponse
{
    [JsonProperty("Id")] public int Id { get; set; }

    [JsonProperty("RoleName")] public string RoleName { get; set; }

    [JsonProperty("RoleDescription")] public string RoleDescription { get; set; }

    [JsonProperty("Members")] public List<MemberRoleReponse> Members { get; set; }
}