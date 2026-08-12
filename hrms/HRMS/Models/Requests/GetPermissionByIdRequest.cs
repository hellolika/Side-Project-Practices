using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class GetPermissionByIdRequest
{
    [JsonProperty("RoleId")]
    public  int RoleId { get; set; }
}