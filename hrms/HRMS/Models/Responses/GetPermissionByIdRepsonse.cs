using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class GetPermissionByIdRepsonse
{
    [JsonProperty("PermissionId")]
    public int PermissionId { get; set; }
}