using Newtonsoft.Json;

namespace HRMS.Models.Responses;

public class PositionResponse
{
    [JsonProperty("Id")]
    public int Id { get; set; }

    [JsonProperty("PositionName")]
    public string PositionName { get; set; }
    
    [JsonProperty("TeamId")]
    public int TeamId { get; set; }
    
    [JsonProperty("TeamName")]
    public string TeamName { get; set; }
}