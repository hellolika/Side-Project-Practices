using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class UpdateRolePermissionRequest
{
    
    [JsonProperty("RoleId")]
    public int RoleId { get; set; }

    [JsonProperty("PermissionList")]
    public List<int> PermissionList { get; set; }

}