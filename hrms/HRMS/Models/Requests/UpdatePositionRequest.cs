using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class UpdatePositionRequest
{
    public int PositionId { get; set; }
    public int TeamId { get; set; }
    public string PositionName { get; set; }
    
    [JsonIgnore]
    public int ModifiedBy { get; set; }
}