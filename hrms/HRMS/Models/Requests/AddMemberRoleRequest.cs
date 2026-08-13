using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class AddMemberRoleRequest
{
    [JsonProperty("MemberId")] public int MemberId { get; set; }

    [JsonProperty("RoleId")] public int RoleId { get; set; }
    
    [JsonIgnore] public int CreatedBy { get; set; }
    
}