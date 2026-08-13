using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class MemberPermissionResponse
{
    [JsonProperty("RoleId")]
    public int RoleId { get; set; }
    
    [JsonProperty("RoleName")]
    public string RoleName { get; set; }
    
    [JsonProperty("PermissionId")]
    public int PermissionId { get; set; }
    
    [JsonProperty("PermissionName")]
    public string PermissionName { get; set; }
}