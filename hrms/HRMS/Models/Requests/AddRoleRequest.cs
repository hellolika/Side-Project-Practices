using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class AddRoleRequest
{
    [JsonProperty("RoleName")] public string RoleName { get; set; }

    [JsonProperty("RoleDescription")] public string RoleDescription { get; set; }
    
    [JsonIgnore] public int CreatedBy { get; set; }
}