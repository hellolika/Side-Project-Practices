CREATE TABLE [dbo].[LocationDetail]
(
	[Id] INT NOT NULL PRIMARY KEY IDENTITY(1, 1),
	[LocationName] NVARCHAR(50) NOT NULL,
	[Latitude] NVARCHAR(50) NOT NULL,
	[Longitude] NVARCHAR(50) NOT NULL,
	[Range] INT DEFAULT 0,
	[IsEnabled] BIT DEFAULT 1,
	[CreatedOn] DATETIME DEFAULT GETDATE(),
	[CreatedBy] INT NULL,
	[ModifiedOn] DATETIME DEFAULT GETDATE(),
	[ModifiedBy] INT NULL
)
