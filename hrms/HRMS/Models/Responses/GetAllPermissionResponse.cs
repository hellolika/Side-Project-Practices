using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class GetAllPermissionResponse
{
    [JsonProperty("Id")]
    public int Id { get; set; }

    [JsonProperty("PermissionName")]
    public string PermissionName { get; set; }
    
    [JsonProperty("PermissionCategoryName")]
    public string PermissionCategoryName { get; set; }
}