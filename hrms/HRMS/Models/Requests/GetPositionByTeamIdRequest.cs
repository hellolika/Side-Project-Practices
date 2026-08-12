using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class GetPositionByTeamIdRequest
{
    [JsonProperty("TeamId")]
    public int TeamId { get; set; }
}