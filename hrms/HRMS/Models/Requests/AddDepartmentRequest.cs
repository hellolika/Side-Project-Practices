using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class AddDepartmentRequest
{
    public string DepartmentName { get; set; }
    
    [JsonIgnore]
    public int CreatedBy { get; set; }
}