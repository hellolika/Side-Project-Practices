using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class AddPositionRequest
{
    public int TeamId { get; set; }
    public string PositionName { get; set; }
    
    [JsonIgnore]
    public int CreatedBy { get; set; }
}